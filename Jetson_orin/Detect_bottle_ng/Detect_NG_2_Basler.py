import socket
import cv2  # opencv library, use to handle image
import numpy as np  # Use to work with image by use array[]
import json
import time
import torch  # Use to check gpu
import platform  # Use to getting the operating system name
import subprocess  # Use to executing a shell command
import threading
import struct  # Use to pack image size into bytes
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
camera2.OffsetX.Value = 440
camera2.OffsetY.Value = 300


camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

torch.cuda.set_device(0)  # Set to your desired GPU number
# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # Bind the socket to a specific IP address and port
ip = "192.168.3.62"
port = 9090
buffer_size = 4096

server_address = (ip, port)
server_socket.bind(server_address)
# Listen for incoming connections
server_socket.listen(1)

is_connection = True

pTime_basler_1 = time.time()
pTime_basler_2 = time.time()
pTime_of_whole_process = 0


host_ip = "192.168.0.9"  # PLCのIPアドレス
host_port = 8501


model_bottleneck = YOLO("/home/orin/Documents/Ultralytic/best_bottleneck_1.engine")
model_body = YOLO("/home/orin/Documents/Ultralytic/best_than_1.engine")


contrast = 1
brightness = 1
accuracy = 0.25


is_start_from_client = False
is_collect_data = False
is_trigger_basler_1 = False
is_trigger_basler_2 = False
is_ng_in_circle = False
is_ok_in_circle = False
is_ping_ok = False
is_stop_because_lost_connect = False


check_result = True
total_ng = 1
total_ok = 1


init_basler_1 = True
init_basler_2 = True


threshold_basler_1 = 9
threshold_basler_2 = 9


# Create a lock object
send_lock = threading.Lock()
plc_host_ip = "192.168.3.10"  
plc_host_port = 8501

cwd = "/media/orin/48C9-C4CB"
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

device_plc_trigger = "1001"
device_plc_total_ok = "1002"
device_plc_total_ng = "1005"
device_plc_run_on = "1007"
device_plc_ng_in_cirle = "1009"
device_plc_start_orin = "1011"


def ping(host):
    param = '-n' if platform.system().lower()=='windows' else '-c'
    command = ['ping', param, '1', host]
    # Redirecting output to DEVNULL to hide it in command line
    return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def empty(a):
    pass


def is_json_empty(json_obj):
    return len(json_obj) == 0


def calculate_ng_area_worker(
    annotated_frame, cropped_image, threshold_value, camera_name, count, x2, y2
):
    finder = CalculateNGArea(cropped_image, threshold_value, 1, camera_name, count)
    _, _, max_distance = finder.get_distance_of_longest_line()
    cv2.putText(
        annotated_frame,
        f"width: {int(max_distance)}",
        (x2 + 10, y2 + 10),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 255),
        3,
    )


def send_image_basler_to_client(frame, fps, processTime, camera):
    global send_lock
    # clock this thread until it done to start anther thread
    with send_lock:
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            raise IOError("Could not encode the basler image as JPG")
        
        image_data = buffer.tobytes()  # Convert the buffer to bytes

        # Prepare to send fps and processTime as floats and the image size as an int
        # f = float
        # I = integer
        # > = big-endian
        header = struct.pack(">IIII", camera, fps, processTime, len(image_data))
        client_socket.sendall(header)  # Send the header

        # Then, send the image data
        client_socket.sendall(image_data)


def send_image_basler_2_to_client(frame, fps, processTime, camera=2):
    global send_lock
    # clock this thread until it done to start anther thread
    with send_lock:
        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            raise IOError("Could not encode the basler image as JPG")

        image_data = buffer.tobytes()  # Convert the buffer to bytes
        # Prepare to send fps and processTime as floats and the image size as an int
        # f = float
        # I = integer 
        # > = big-endian
        header = struct.pack(">IIII", camera, fps, processTime, len(image_data))
        client_socket.sendall(header)  # Send the header
        
        # Then, send the image data
        client_socket.sendall(image_data)


def capture_and_process_images_from_basler_1():
    try:
        global is_start_from_client, is_collect_data
        global pTime_basler_1
        global accuracy, contrast, brightness
        global cwd, is_ng_in_circle, is_ok_in_circle, total_ng, total_ok
        global send_lock
        global init_basler_1, is_stop_because_lost_connect
        global check_result
        global conn
        
        while True:
            with send_lock:
                grabResult1 = camera1.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException
                )

                is_running = conn.read_data_from_plc("DM", device_plc_run_on, ".D", 1)
                
                if grabResult1.GrabSucceeded():

                    current_datetime = datetime.now()
                    today = current_datetime.strftime("%y-%m-%d")
                    month = current_datetime.strftime("%B")
                    year = current_datetime.year
                    current_time = current_datetime.strftime("%H-%M-%S")

                    # checking time
                    curr_time = time.time()
                    processTime = curr_time - pTime_basler_1
                    fps = 1 / (
                        curr_time - pTime_basler_1
                    )  # tính fps (Frames Per Second) - đây là chỉ số khung hình trên mỗi giây
                    pTime_basler_1 = curr_time

                    # print(is_running)

                    if "1" in str(is_running) or init_basler_1 == True:
                        if is_stop_because_lost_connect == False:
                            # print("in detect b")
                            check_result = True
                            if init_basler_1 == False:
                                is_ok_in_circle = True
                                
                            image = converter.Convert(grabResult1)
                            img = image.GetArray()
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            result = model_bottleneck.predict(
                                source=img,
                                device=0,
                                conf=0.5,
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
                                if is_ng_in_circle == False:
                                    if init_basler_1 == False:
                                        # print(f"total ng: {total_ng}")
                                        print("basler 1 detected ng")
                                        conn.send_data_to_plc("DM", device_plc_total_ng, ".D", total_ng)
                                        conn.send_data_to_plc("DM", device_plc_ng_in_cirle, ".D", 1)
                                        total_ng += 1
                                        is_ng_in_circle = True

                                is_ng = True
                                result_color = (0, 0, 255)
                                result_message = "NG"
                                
                                #is_ok = False

                                # count = 0
                                # for parsed_data in parsed_datas:
                                #     box_info = parsed_data["box"]
                                #     x1 = int(box_info["x1"])
                                #     y1 = int(box_info["y1"])
                                #     x2 = int(box_info["x2"])
                                #     y2 = int(box_info["y2"])
                                #     # print("x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2)

                                #     # Crop the image to the bounding box
                                #     cropped_image = img[y1:y2, x1:x2]
                                #     cropped_image = cv2.resize(cropped_image, (300, 300))

                                #     thread = threading.Thread(
                                #         target=calculate_ng_area_worker(
                                #             annotated_frame,
                                #             cropped_image,
                                #             threshold_value,
                                #             "Basler 1",
                                #             count,
                                #             x2,
                                #             y2,
                                #         ),
                                #     )
                                #     thread.start()
                                #     count += 1

                            directory1 = cwd + f"/Data_storage/Bottle_neck"
                            directory2 = cwd + f"/Data_storage/Bottle_neck/{str(year)}"
                            directory3 = cwd + f"/Data_storage/Bottle_neck/{str(year)}/{str(month)}"
                            directory4 = cwd + f"/Data_storage/Bottle_neck/{str(year)}/{str(month)}/{str(today)}"
                            directory5 = cwd + f"/Data_storage/Bottle_neck/{str(year)}/{str(month)}/{str(today)}/ImageError"

                            if not os.path.isdir(directory1):
                                os.mkdir(directory1)
                            if not os.path.isdir(directory2):
                                os.mkdir(directory2)
                            if not os.path.isdir(directory3):
                                os.mkdir(directory3)
                            if not os.path.isdir(directory4):
                                os.mkdir(directory4)
                            if not os.path.isdir(directory5):
                                os.mkdir(directory5)

                            cv2.putText(
                                annotated_frame,
                                f"FPS: {int(fps)}",
                                (0, 100),
                                cv2.FONT_HERSHEY_PLAIN,
                                3,
                                (255, 255, 0),
                                3,
                            )

                            cv2.putText(
                                annotated_frame,
                                result_message,
                                (0, 200),
                                cv2.FONT_HERSHEY_PLAIN,
                                3,
                                result_color,
                                3,
                            )

                            annotated_frame = cv2.resize(
                                annotated_frame, (800, 700)
                            )  # Adjust the width and height as desiredPIR

                            # annotated_frame = cv2.resize(
                            #     annotated_frame, (640, 480)
                            # )
                            parsed_processTime = int(processTime * 1000)
                            if is_start_from_client:
                                send_image_basler_to_client(annotated_frame, int(fps), parsed_processTime, 1)

                            elif is_collect_data:
                                img = cv2.resize(img, (800, 700))  # Adjust the width and height as desiredPIR
                                send_image_basler_to_client(img, int(fps), parsed_processTime, 1)

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
        global is_start_from_client, is_collect_data
        global pTime_basler_2, is_trigger_basler_2
        global accuracy, contrast, brightness
        global cwd, is_ng_in_circle, total_ng, total_ok
        global send_lock
        global init_basler_2, is_ok_in_circle
        global check_result , is_stop_because_lost_connect
        global conn

        while True:
            with send_lock:
                
                grabResult2 = camera2.RetrieveResult(
                    5000, pylon.TimeoutHandling_ThrowException
                )
                is_running = conn.read_data_from_plc("DM", device_plc_run_on, ".D", 1)

                if grabResult2.GrabSucceeded():

                    current_datetime = datetime.now()
                    today = current_datetime.strftime("%y-%m-%d")
                    month = current_datetime.strftime("%B")
                    year = current_datetime.year
                    current_time = current_datetime.strftime("%H-%M-%S")

                    # checking time
                    curr_time = time.time()
                    processTime = curr_time - pTime_basler_2
                    fps = 1 / (
                        curr_time - pTime_basler_2
                    )  # tính fps (Frames Per Second) - đây là chi số khung hình trên mỗi giây
                    pTime_basler_2 = curr_time

                    # print(is_running)

                    if "1" in str(is_running) or init_basler_2 == True:
                        if is_stop_because_lost_connect == False:
                            check_result = True
                            if init_basler_2 == False:
                                is_ok_in_circle = True
                                
                            is_ready_to_send = True
                            image = converter.Convert(grabResult2)
                            img = image.GetArray()
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                            # Access the image data
                            result = model_body.predict(
                                source=img,
                                device=0,
                                conf=0.5,
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
                                if is_ng_in_circle == False:
                                    print("basler 2 detected ng")
                                    if init_basler_2 == False:
                                        conn.send_data_to_plc("DM", device_plc_total_ng, ".D", total_ng)
                                        conn.send_data_to_plc("DM", device_plc_ng_in_cirle, ".D", 1)
                                        total_ng += 1
                                        is_ng_in_circle = True

                                result_color = (0, 0, 255)
                                result_message = "NG"

                                # count = 0
                                # for parsed_data in parsed_datas:
                                #     box_info = parsed_data["box"]
                                #     x1 = int(box_info["x1"])
                                #     y1 = int(box_info["y1"])
                                #     x2 = int(box_info["x2"])
                                #     y2 = int(box_info["y2"])
                                #     # print("x1:", x1, "y1:", y1, "x2:", x2, "y2:", y2)

                                #     # Crop the image to the bounding box
                                #     cropped_image = img[y1:y2, x1:x2]
                                #     cropped_image = cv2.resize(cropped_image, (300, 300))

                                #     thread = threading.Thread(
                                #         target=calculate_ng_area_worker(
                                #             annotated_frame,
                                #             cropped_image,
                                #             threshold_value,
                                #             "Basler 2",
                                #             count,
                                #             x2,
                                #             y2,
                                #         ),
                                #     )
                                #     thread.start()
                                #     count += 1

                            directory1 = cwd + f"/Data_storage/Body"
                            directory2 = cwd + f"/Data_storage/Body/{str(year)}"
                            directory3 = cwd + f"/Data_storage/Body/{str(year)}/{str(month)}"
                            directory4 = cwd + f"/Data_storage/Body/{str(year)}/{str(month)}/{str(today)}"
                            directory5 = cwd + f"/Data_storage/Body/{str(year)}/{str(month)}/{str(today)}/ImageError"

                            if not os.path.isdir(directory1):
                                os.mkdir(directory1)
                            if not os.path.isdir(directory2):
                                os.mkdir(directory2)
                            if not os.path.isdir(directory3):
                                os.mkdir(directory3)
                            if not os.path.isdir(directory4):
                                os.mkdir(directory4)
                            if not os.path.isdir(directory5):
                                    os.mkdir(directory5)

                            cv2.putText(
                                annotated_frame,
                                f"FPS: {int(fps)}",
                                (0, 100),
                                cv2.FONT_HERSHEY_PLAIN,
                                5,
                                (255, 255, 0),
                                5,
                            )

                            cv2.putText(
                                annotated_frame,
                                result_message,
                                (0, 200),
                                cv2.FONT_HERSHEY_PLAIN,
                                3,
                                result_color,
                                3,
                            )

                            annotated_frame = cv2.resize(
                                annotated_frame, (700, 1000)
                            )  # Adjust the width and height as desired

                            # annotated_frame = cv2.resize(
                            #     annotated_frame, (640, 480)
                            # )
                            # current_time = current_datetime.strftime("%H-%M-%S-%f")
                            # try:
                            #     print(f"take a picture {current_time}")
                            #     cv2.imwrite(f"{directory5}/NG-{current_time}.jpg", annotated_frame)
                            # except Exception as e:
                            #     print(e)

                            parsed_processTime = int(processTime * 1000)
                            if is_start_from_client: 
                                send_image_basler_to_client(annotated_frame, int(fps), parsed_processTime, 2)

                            elif is_collect_data:
                                img = cv2.resize(img, (700, 1000))  # Adjust the width and height as desiredPIR
                                send_image_basler_to_client(img, int(fps), parsed_processTime, 1)
                                
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
        print("Thread of function basler_2() get some error: ", e)

def cirle_detect_result():
    global send_lock, is_ng_in_circle, is_ok_in_circle
    global total_ok, total_ng, init_basler_1, init_basler_2
    global check_result, is_stop_because_lost_connect
    global conn

    is_running = conn.read_data_from_plc("DM", device_plc_run_on, ".D", 1)
    if is_stop_because_lost_connect == False:
        if "1" not in is_running and is_ng_in_circle == True and check_result == True:
            is_ng_in_circle = False
            check_result = False
            print("reset circle ng")

        if "1" not in is_running and is_ng_in_circle == False and is_ok_in_circle == True and check_result == True:
            conn.send_data_to_plc("DM", device_plc_total_ok, ".D", total_ok)
            total_ok+= 1
            print("reset circle ok")
            print(total_ok)
            is_ok_in_circle = False
            check_result = False


def check_connect_to_plc():
    global send_lock, client
    global is_stop_because_lost_connect
    while True:
        try:
            time.sleep(1)
            check = ping("192.168.3.10")
            if check:
                if is_stop_because_lost_connect == True:
                    is_stop_because_lost_connect = False
                    client.close()
                    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    conn = PLCConnect(client, plc_host_ip, plc_host_port)

                    conn.connect()
                    print("connection lost... reconnecting" ) 
                    time.sleep(1)
                    print("reconnected")
            else:
                print("can not ping")
                is_stop_because_lost_connect = True


        except Exception as e:
            print(e)


def get_command_from_client():

    global client_socket, client_address, buffer_size
    global is_start_from_client, is_collect_data
    global is_trigger_basler_1, is_trigger_basler_2
    global accuracy, ip, port, is_connection


    print(f"Server Start At {ip}:{port}")
    print('Waiting for a connection...')

    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print(f'Client IP: {client_address} connected')
    print("Waiting for command")

    while True:
        try:
            data = client_socket.recv(buffer_size)              
            data_response = data.decode()
            print(data_response)

            if data_response == "Start Camera":
                is_start_from_client = True
                is_collect_data = False

            elif data_response == "Stop Camera":
                is_start_from_client = False

            elif "Change Accuracy" in data_response:
                accuracy_parsed = int(data_response.split('-')[1]) / 100
                accuracy = accuracy_parsed
                print(accuracy_parsed)

            elif "Ping PLC" in data_response:
                ip = data_response.split('-')
                if ping(ip):
                    print("ping ok")

            elif "Disconnect" in data_response:
                is_start_from_client = False
                client_socket.close()
                print(f'Client IP: {client_address} has been disconnected')
                print('Waiting for a connection...')

                # Accept a connection
                client_socket, client_address = server_socket.accept()
                print(f'Client IP: {client_address} connected')
                print("Waiting for command")

            elif "Continue Trigger" in data_response:
                is_trigger_basler_1 = True
                is_trigger_basler_2 = False

            elif "Start Collect Data" in data_response:
                is_start_from_client = False
                is_collect_data = True

        except Exception as e:
            print("Thread of function get_command_from_client() get some error: ", e)
            is_connection = False

            server_socket.listen(1)
            print('Waiting for a connection again...')

            # Accept a connectio
            client_socket, client_address = server_socket.accept()
            print(f'Client IP: {client_address} connected')

def program_close():
    print("Script is stopping, cleanup...")


def main():

    conn.connect()
    conn.send_data_to_plc("DM", device_plc_start_orin, ".D", 1)

    # Setup threads
    thread_1 = threading.Thread(target=capture_and_process_images_from_basler_1)
    thread_2 = threading.Thread(target=capture_and_process_images_from_basler_2)
    thread_3 = threading.Thread(target=check_connect_to_plc)
    thread_4 = threading.Thread(target=get_command_from_client)

    # Start threads
    thread_1.start()
    thread_2.start()
    thread_3.start()
    thread_4.start()

    # Wait for threads to complete
    thread_1.join()
    thread_2.join()
    thread_3.join()
    # thread_4.join()

    # Clean up resources
    camera1.StopGrabbing()
    camera2.StartGrabbing()
    atexit.register(program_close)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
