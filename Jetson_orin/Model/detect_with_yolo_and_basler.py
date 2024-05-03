import cv2
import numpy as np
import json
import time
import torch

from ultralytics import YOLO
from pypylon import pylon

torch.cuda.set_device(0)  # Set to your desired GPU number

# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


pTime = 0

model = YOLO("/home/orin/Documents/Ultralytic/best_45_500.engine")

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        # Resize the image
        resized_img = cv2.resize(
            img, (2550, 1900)
        )  # Adjust the width and height as desiredPIR

        # checking time
        curr_time = time.time()

        fps = 1 / (
            curr_time - pTime
        )  # tính fps (Frames Per Second) - đây là chỉ số khung hình trên mỗi giây

        print(f"test : {curr_time - pTime}")

        pTime = curr_time

        try:
            result = model.predict(source=resized_img, device=0, conf=0.25, iou=0.7)
            jsonResult = result[0].tojson()
            parsed_data = json.loads(jsonResult)
            annotated_frame = result[0].plot()

            cv2.putText(
                annotated_frame,
                f"FPS: {int(fps)}",
                (0, 50),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 255, 0),
                3,
            )

            # Resize the image
            annotated_frame = cv2.resize(annotated_frame, (1300, 1000))  # Adjust the width and height as desiredPIR


            cv2.imshow("NG", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except Exception as e:
            print(e)

    grabResult.Release()

# Releasing the resource
camera.StopGrabbing()

cv2.destroyAllWindows()