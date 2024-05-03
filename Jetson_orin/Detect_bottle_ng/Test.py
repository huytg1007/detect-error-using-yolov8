import socket
import cv2  # opencv library, use to handle image
import numpy as np  # Use to work with image by use array[]
import json
import time
import torch  # Use to check gpu
import platform  # Use to getting the operating system name
import subprocess  # Use to executing a shell command
import threading
import atexit  # Use to check exit
import os

from pypylon import pylon
from datetime import datetime, date
from Utility import CalculateNGArea
from ultralytics import YOLO
from PLC_connect import PLCConnect

# Instantiate the pylon DeviceInfo object and use it to get the cameras
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
if len(devices) < 2:
    raise ValueError("Not enough cameras found")

# Create two camera objects
camera2 = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
camera1 = pylon.InstantCamera(tl_factory.CreateDevice(devices[1]))


# Open the cameras
camera1.Open()
camera2.Open()


camera1.Width.Value = 1900
camera1.Height.Value = 1500
camera1.OffsetX.Value = 370
camera1.OffsetY.Value = 440


camera2.Width.Value = 1900
camera2.Height.Value = 1500
camera2.OffsetX.Value = 350
camera2.OffsetY.Value = 300


camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


pTime_basler_1 = time.time()
pTime_basler_2 = time.time()
pTime_of_whole_process = 0


model_bottleneck = YOLO("/home/orin/Documents/Ultralytic/best_co_chai.engine")
model_body = YOLO("/home/orin/Documents/Ultralytic/best_than_1.engine")


contrast = 1
brightness = 1
accuracy = 0.6


is_ng_in_circle = False
is_ok_in_circle = False


is_ng_in_body = False
is_ng_in_bottleneck = False


is_stop_because_lost_connect = False
check_result_of_one_circle = True  # Check to send only 1 time ng or ok each circle


is_any_ng_bigger_than_accept_camera_1 = True
is_any_ng_bigger_than_accept_camera_2 = True


init_basler_1 = True  # initial balser 1 to make sure model is loaded
init_basler_2 = True  # initial balser 2 to make sure model is loaded


total_ng = 0
total_ok = 0

total_body_ng = 0
total_body_ok = 0

total_bottle_neck_ng = 0
total_bottle_neck_ok = 0

threshold_basler_1 = 9
threshold_basler_2 = 9

distance_body = 13
distance_bottleneck = 70


# Create a lock object
send_lock = threading.Lock()

plc_host_ip = "192.0.9.123"
plc_host_port = 8501

cwd = "/media/orin/48C9-C4CB1"
# cwd = "/home/orin/Documents/Detect_bottle_ng"

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
conn = PLCConnect(client, plc_host_ip, plc_host_port)

# PLC memery
# 1001 trigger
# 1002 total ok
# 1005 total ng
# 1007 run on
# 1009 send to 1 if ng in cirle
# 1011 send to 1 if start success
# 1013 send to 1 if ng and 2 if success after 1 circle it will return to 0

# 1015 total body ok
# 1017 total body ng
# 1019 total bottle_neck ok
# 1021 total bottle_neck ng


device_plc_trigger = "1001"
device_plc_total_ok = "1002"
device_plc_total_ng = "1005"
device_plc_run_on = "1007"
device_plc_ng_in_cirle = "1009"
device_plc_start_orin = "1011"
device_plc_send_result = "1013"

device_plc_body_ok = "1015"
device_plc_body_ng = "1017"
device_plc_bottleneck_ok = "1019"
device_plc_bottleneck_ng = "1021"


conn.send_data_to_plc("DM", device_plc_send_result, ".D", 0)


def ping(host):
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, "1", host]
    # Redirecting output to DEVNULL to hide it in command line
    return (
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        == 0
    )


def empty(a):
    pass


def is_json_empty(json_obj):
    # return true if length is 0.
    return len(json_obj) == 0


# THIS FUNC USE TO CHECK THE NG THAT DECTECT IS > MIN ACCEPT
def calculate_ng_area_worker(
    annotated_frame, cropped_image, threshold_value, camera_name, res, count, x2, y2
):
    global is_any_ng_bigger_than_accept_camera_1, is_any_ng_bigger_than_accept_camera_2
    global distance_body, distance_bottleneck
    finder = CalculateNGArea(cropped_image, threshold_value, 1, camera_name, res, count)
    _, _, distance = finder.show_result()

    if camera_name == "Basler 1" and distance > distance_bottleneck:
        is_any_ng_bigger_than_accept_camera_1 = True
        # print(f"Basler 1 have ng {distance}")

    if camera_name == "Basler 2" and distance > distance_body and distance < 40:
        is_any_ng_bigger_than_accept_camera_2 = True
        print(f"Basler 2 have ng with distance {distance}")

    cv2.putText(
        annotated_frame,
        f"w: {int(distance)}",
        (x2 + 10, y2 + 10),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 255),
        3,
    )


# THIS FUNC WILL RUN ON A THREAD TO IMPROVE PERFORMANCE
def capture_and_process_images_from_basler_1():
    try:
        global accuracy, contrast, brightness
        global init_basler_1, is_stop_because_lost_connect
        global pTime_basler_1, is_any_ng_bigger_than_accept_camera_1
        global total_ng, total_ok, total_bottle_neck_ng, total_bottle_neck_ok, total_body_ng
        global is_ng_in_circle, is_ng_in_bottleneck
        global send_lock, conn, cwd, is_ok_in_circle
        global check_result_of_one_circle
        global distance_bottleneck

        cv2.namedWindow("Parameters", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("Parameters", 300, 150)
        cv2.createTrackbar("Width", "Parameters", 70, 100, empty)  # type: ignore
        cv2.createTrackbar("Threshold", "Parameters", 30, 50, empty)  # type: ignore
        cv2.createTrackbar("Res", "Parameters", 20, 40, empty)  # type: ignore
        cv2.createTrackbar("Bright", "Parameters", 40, 100, empty)  # type: ignore
        cv2.createTrackbar("Accuracy", "Parameters", 60, 100, empty)  # type: ignore

        while True:
            with send_lock:

                grabResult1 = camera1.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException
                )

                is_running = conn.read_data_from_plc("DM", device_plc_run_on, ".D", 1)

                if grabResult1.GrabSucceeded():

                    distance_bottleneck = cv2.getTrackbarPos("Width", "Parameters")
                    if distance_bottleneck < 1:
                        distance_bottleneck = 1

                    threshold_value = cv2.getTrackbarPos("Threshold", "Parameters")
                    if threshold_value < 9:
                        threshold_value = 9

                    res_value = cv2.getTrackbarPos("Res", "Parameters")
                    if res_value < 10:
                        res_value = 10

                    accuracy_value = cv2.getTrackbarPos("Accuracy", "Parameters")
                    if accuracy_value < 25:
                        accuracy_value = 25

                    accuracy_value = accuracy_value / 100
                    brightness_value = cv2.getTrackbarPos("Bright", "Parameters")

                    current_datetime = datetime.now()
                    today = current_datetime.strftime("%y-%m-%d")
                    month = current_datetime.strftime("%B")
                    year = current_datetime.year
                    current_time = current_datetime.strftime("%H-%M-%S")

                    # checking time
                    curr_time = time.time()
                    fps = 1 / (
                        curr_time - pTime_basler_1
                    )  # tính fps (Frames Per Second) - đây là chỉ số khung hình trên mỗi giây
                    pTime_basler_1 = curr_time

                    # print(is_running)
                    # Initial program because yolo model need time to load
                    # Only run yolo detect when get signal 1 on PLC
                    if "1" in str(is_running) or init_basler_1 == True:
                        # Check if lost connect
                        if is_stop_because_lost_connect == False:
                            # check to
                            check_result_of_one_circle = True
                            if init_basler_1 == False:
                                is_ok_in_circle = True

                            image = converter.Convert(grabResult1)
                            img = image.GetArray()
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                            img = cv2.addWeighted(img, 1, img, 0, brightness_value)
                            result = model_bottleneck.predict(
                                source=img,
                                device=0,
                                conf=accuracy_value,
                                iou=0.6,
                                verbose=False,
                            )
                            jsonResult = result[0].tojson()
                            parsed_datas = json.loads(jsonResult)
                            annotated_frame = result[0].plot()

                            result_message = ""
                            result_color = (255, 255, 0)

                            if is_json_empty(parsed_datas):
                                result_color = (0, 128, 0)
                                result_message = "OK"

                            else:
                                count = 0
                                for parsed_data in parsed_datas:
                                    box_info = parsed_data["box"]
                                    x1 = int(box_info["x1"])
                                    y1 = int(box_info["y1"])
                                    x2 = int(box_info["x2"])
                                    y2 = int(box_info["y2"])
                                    # print("x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2)

                                    # Crop the image to the bounding box
                                    cropped_image = img[y1:y2, x1:x2]

                                    width, height, _ = cropped_image.shape

                                    cropped_image = cv2.resize(
                                        cropped_image, (width * 30, height * 30)
                                    )

                                    thread = threading.Thread(
                                        target=calculate_ng_area_worker(
                                            annotated_frame,
                                            cropped_image,
                                            threshold_value,
                                            "Basler 1",
                                            res_value,
                                            count,
                                            x2,
                                            y2,
                                        ),
                                    )
                                    thread.start()
                                    count += 1

                                if (
                                    is_ng_in_circle == False
                                    and is_any_ng_bigger_than_accept_camera_1 == True
                                ):
                                    if init_basler_1 == False:

                                        print("basler 1 detected ng")

                                        is_ng_in_circle = True
                                        total_ng += 1
                                        total_bottle_neck_ng += 1

                                        directory1 = cwd + f"/Data_storage/Bottle_neck"
                                        directory2 = (
                                            cwd
                                            + f"/Data_storage/Bottle_neck/{str(year)}_{str(month)}_{str(today)}_ImageError"
                                        )
                                        if not os.path.isdir(directory1):
                                            os.mkdir(directory1)
                                        if not os.path.isdir(directory2):
                                            os.mkdir(directory2)

                                        conn.send_data_to_plc(
                                            "DM",
                                            device_plc_bottleneck_ng,
                                            ".D",
                                            total_bottle_neck_ng,
                                        )
                                        conn.send_data_to_plc(
                                            "DM", device_plc_send_result, ".D", 1
                                        )
                                        conn.send_data_to_plc(
                                            "DM", device_plc_total_ng, ".D", total_ng
                                        )
                                        conn.send_data_to_plc(
                                            "DM", device_plc_ng_in_cirle, ".D", 1
                                        )

                                        # if is_ng_in_circle == False:
                                        #     is_ng_in_circle = True
                                        #     conn.send_data_to_plc("DM", device_plc_total_ng, ".D", total_ng)
                                        #     conn.send_data_to_plc("DM", device_plc_ng_in_cirle, ".D", 1)

                                        current_time = current_datetime.strftime(
                                            "%H-%M-%S-%f"
                                        )
                                        try:
                                            print(
                                                f"take a ng picture at {current_time}"
                                            )
                                            cv2.imwrite(
                                                f"{directory2}/NG-{current_time}.jpg",
                                                annotated_frame,
                                            )
                                        except Exception as e:
                                            print(e)

                            cv2.putText(
                                annotated_frame,
                                f"FPS: {int(fps)}",
                                (0, 50),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (255, 255, 0),
                                2,
                            )

                            cv2.putText(
                                annotated_frame,
                                f"Total: {total_ng + total_ok}",
                                (0, 100),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (255, 255, 0),
                                2,
                            )

                            cv2.putText(
                                annotated_frame,
                                f"Total bottleneck ng: {total_bottle_neck_ng}",
                                (0, 150),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (0, 0, 255),
                                2,
                            )

                            cv2.putText(
                                annotated_frame,
                                f"Total ng: {total_ng}",
                                (0, 200),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (0, 0, 255),
                                2,
                            )

                            annotated_frame = cv2.resize(annotated_frame, (600, 700))
                            cv2.imshow("Basler 1 Camera", annotated_frame)

                    else:
                        cirle_detect_result()

                    if init_basler_1 == True:
                        init_basler_1 = False

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                else:
                    print("camera 1: ", grabResult1.GrabSucceeded())

            grabResult1.Release()

    except Exception as e:
        print(
            "Thread of function capture_and_process_images_from_basler_1() get some error: ",
            e,
        )


def capture_and_process_images_from_basler_2():
    try:
        global accuracy, contrast, brightness
        global init_basler_2, is_stop_because_lost_connect
        global pTime_basler_2, is_any_ng_bigger_than_accept_camera_2
        global total_ng, total_ok, total_body_ng, total_body_ok, total_bottle_neck_ng
        global is_ng_in_circle, is_ng_in_body
        global send_lock, conn, cwd, is_ok_in_circle, check_result_of_one_circle
        global distance_body

        cv2.namedWindow("Parameters_2", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("Parameters_2", 300, 150)
        cv2.createTrackbar("Width", "Parameters_2", 13, 100, empty)  # type: ignore
        cv2.createTrackbar("Threshold", "Parameters_2", 40, 50, empty)  # type: ignore
        cv2.createTrackbar("Res", "Parameters_2", 30, 40, empty)  # type: ignore
        cv2.createTrackbar("Bright", "Parameters_2", 6, 100, empty)  # type: ignore
        cv2.createTrackbar("Accuracy", "Parameters_2", 40, 100, empty)  # type: ignore

        while True:
            with send_lock:

                grabResult2 = camera2.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException
                )
                is_running = conn.read_data_from_plc("DM", device_plc_run_on, ".D", 1)
                if grabResult2.GrabSucceeded():

                    distance_body = cv2.getTrackbarPos("Width", "Parameters_2")
                    if distance_body < 1:
                        distance_body = 1

                    threshold_value = cv2.getTrackbarPos("Threshold", "Parameters_2")
                    if threshold_value < 9:
                        threshold_value = 9

                    res_value = cv2.getTrackbarPos("Res", "Parameters_2")
                    if res_value < 10:
                        res_value = 10

                    accuracy_value = cv2.getTrackbarPos("Accuracy", "Parameters_2")
                    if accuracy_value < 25:
                        accuracy_value = 25

                    accuracy_value = accuracy_value / 100

                    brightness_value = cv2.getTrackbarPos("Bright", "Parameters_2")

                    current_datetime = datetime.now()
                    today = current_datetime.strftime("%y-%m-%d")
                    month = current_datetime.strftime("%B")
                    year = current_datetime.year
                    current_time = current_datetime.strftime("%H-%M-%S")

                    # checking time
                    curr_time = time.time()
                    fps = 1 / (
                        curr_time - pTime_basler_2
                    )  # tính fps (Frames Per Second) - đây là chi số khung hình trên mỗi giây
                    pTime_basler_2 = curr_time
                    # print(is_running)

                    if "1" in str(is_running) or init_basler_2 == True:
                        if is_stop_because_lost_connect == False:
                            check_result_of_one_circle = True
                            if init_basler_2 == False:
                                is_ok_in_circle = True

                            image = converter.Convert(grabResult2)
                            img = image.GetArray()
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                            img = cv2.addWeighted(img, 1, img, 0, brightness_value)

                            # Access the image data
                            result = model_body.predict(
                                source=img,
                                device=0,
                                conf=accuracy_value,
                                iou=0.6,
                                verbose=False,
                            )
                            jsonResult = result[0].tojson()
                            parsed_datas = json.loads(jsonResult)
                            annotated_frame = result[0].plot()

                            result_message = ""
                            result_color = (255, 255, 0)

                            if is_json_empty(parsed_datas):
                                result_color = (0, 128, 0)
                                result_message = "OK"

                            else:
                                count = 0
                                for parsed_data in parsed_datas:
                                    box_info = parsed_data["box"]
                                    x1 = int(box_info["x1"])
                                    y1 = int(box_info["y1"])
                                    x2 = int(box_info["x2"])
                                    y2 = int(box_info["y2"])
                                    # print("x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2)

                                    # Crop the image to the bounding box
                                    cropped_image = img[y1:y2, x1:x2]
                                    cropped_image = cv2.resize(
                                        cropped_image, (300, 300)
                                    )

                                    thread = threading.Thread(
                                        target=calculate_ng_area_worker(
                                            annotated_frame,
                                            cropped_image,
                                            threshold_value,
                                            "Basler 2",
                                            res_value,
                                            count,
                                            x2,
                                            y2,
                                        ),
                                    )

                                    thread.start()
                                    count += 1

                                if (
                                    is_ng_in_circle == False
                                    and is_any_ng_bigger_than_accept_camera_2 == True
                                ):
                                    if init_basler_2 == False:
                                        print("basler 2 detected ng")

                                        total_ng += 1
                                        total_body_ng += 1
                                        is_ng_in_circle = True

                                        directory1 = cwd + f"/Data_storage/Body"
                                        directory2 = (
                                            cwd
                                            + f"/Data_storage/Body/{str(year)}_{str(month)}_{str(today)}_ImageError"
                                        )

                                        if not os.path.isdir(directory1):
                                            os.mkdir(directory1)
                                        if not os.path.isdir(directory2):
                                            os.mkdir(directory2)

                                        conn.send_data_to_plc(
                                            "DM",
                                            device_plc_body_ng,
                                            ".D",
                                            total_body_ng,
                                        )
                                        conn.send_data_to_plc(
                                            "DM", device_plc_send_result, ".D", 1
                                        )
                                        conn.send_data_to_plc(
                                            "DM", device_plc_total_ng, ".D", total_ng
                                        )
                                        conn.send_data_to_plc(
                                            "DM", device_plc_ng_in_cirle, ".D", 1
                                        )

                                        # if is_ng_in_circle == False:
                                        #     is_ng_in_circle = True
                                        #     conn.send_data_to_plc("DM", device_plc_total_ng, ".D", total_ng)
                                        #     conn.send_data_to_plc("DM", device_plc_ng_in_cirle, ".D", 1)

                                        current_time = current_datetime.strftime(
                                            "%H-%M-%S-%f"
                                        )
                                        try:
                                            print(
                                                f"take a ng picture at {current_time}"
                                            )
                                            cv2.imwrite(
                                                f"{directory2}/NG-{current_time}.jpg",
                                                annotated_frame,
                                            )
                                        except Exception as e:
                                            print(e)

                            cv2.putText(
                                annotated_frame,
                                f"FPS: {int(fps)}",
                                (0, 50),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (255, 255, 0),
                                2,
                            )

                            cv2.putText(
                                annotated_frame,
                                f"Total: {total_ng + total_ok}",
                                (0, 100),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (255, 255, 0),
                                2,
                            )

                            cv2.putText(
                                annotated_frame,
                                f"Total body ng: {total_body_ng}",
                                (0, 150),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (0, 0, 255),
                                2,
                            )

                            cv2.putText(
                                annotated_frame,
                                f"Total ng: {total_ng}",
                                (0, 200),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,
                                (0, 0, 255),
                                2,
                            )

                            annotated_frame = cv2.resize(annotated_frame, (500, 900))
                            cv2.imshow("Basler 2 Camera", annotated_frame)

                    else:
                        cirle_detect_result()

                    if init_basler_2 == True:
                        init_basler_2 = False

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                else:
                    print("camera 2: ", grabResult2.GrabSucceeded())
            grabResult2.Release()

    except Exception as e:
        print(
            "Thread of function basler_2() get some error: ",
            e,
        )


def cirle_detect_result():
    global is_ng_in_bottleneck, is_ng_in_body
    global send_lock, is_ng_in_circle, is_ok_in_circle, conn
    global total_ok, total_ng, init_basler_1, init_basler_2
    global check_result_of_one_circle, is_stop_because_lost_connect
    global is_any_ng_bigger_than_accept_camera_1, is_any_ng_bigger_than_accept_camera_2

    is_running = conn.read_data_from_plc("DM", device_plc_run_on, ".D", 1)
    if is_stop_because_lost_connect == False:
        if (
            "1" not in is_running
            and is_ng_in_circle == True
            and check_result_of_one_circle == True
            and (
                is_any_ng_bigger_than_accept_camera_1 == True
                or is_any_ng_bigger_than_accept_camera_2 == True
            )
        ):
            is_ng_in_circle = False
            is_ng_in_bottleneck = False
            is_ng_in_body = False
            check_result_of_one_circle = False
            is_any_ng_bigger_than_accept_camera_1 = False
            is_any_ng_bigger_than_accept_camera_2 = False
            print("reset circle ng")
            print("")

        if (
            "1" not in is_running
            and is_ng_in_circle == False
            and is_ok_in_circle == True
            and check_result_of_one_circle == True
        ):
            conn.send_data_to_plc("DM", device_plc_total_ok, ".D", total_ok)
            conn.send_data_to_plc("DM", device_plc_send_result, ".D", 2)
            total_ok += 1
            is_ok_in_circle = False
            check_result_of_one_circle = False
            print("reset circle ok")
            print("")
            time.sleep(0.2)

        conn.send_data_to_plc("DM", device_plc_send_result, ".D", 0)


def check_connect_to_plc():
    global send_lock, client, conn
    global is_stop_because_lost_connect
    while True:
        try:
            time.sleep(1)
            check = ping("192.0.9.123")
            if check:
                if is_stop_because_lost_connect == True:
                    is_stop_because_lost_connect = False

                    client.close()
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    conn = PLCConnect(client, plc_host_ip, plc_host_port)

                    conn.connect()
                    print("connection lost... reconnecting")
                    time.sleep(1)
                    print("reconnected")

            else:
                print("can not ping")
                is_stop_because_lost_connect = True

        except Exception as e:
            print(e)


def program_close():
    print("Script is stopping, cleanup...")


def send_detect_program_is_ruuning_to_plc():
    conn.send_data_to_plc("DM", device_plc_start_orin, ".D", 1)


def main():
    conn.connect()
    # Setup threads
    thread_1 = threading.Thread(target=capture_and_process_images_from_basler_1)
    thread_2 = threading.Thread(target=capture_and_process_images_from_basler_2)
    thread_3 = threading.Thread(target=check_connect_to_plc)
    thread_4 = threading.Thread(target=send_detect_program_is_ruuning_to_plc)

    # Start threads
    thread_1.start()
    thread_2.start()
    thread_3.start()
    thread_4.start()

    # Wait for threads to complete
    thread_1.join()
    thread_2.join()
    # thread_3.join()

    # Clean up resources
    camera1.StopGrabbing()
    camera2.StartGrabbing()
    cv2.destroyAllWindows()
    atexit.register(program_close)


if __name__ == "__main__":
    main()
