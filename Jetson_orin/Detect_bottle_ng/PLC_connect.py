import socket
import time


class PLCConnect:
    def __init__(self, client, host, port):
        self.client = client
        self.host = host
        self.port = port

    def connect(self):
        try:
            # クライアント接続
            print(f"{self.host} {self.port}")
            self.client.connect(
                (self.host, self.port)
            )  # サーバーに接続(kv-7500にTCP接続/上位リンク通信)
            print("PLC Connected")
        except Exception as ex:
            print("PLC接続NG" + str(ex))
            return "PLC接続NG"
        return "Connected"

    def disconnect(self):
        self.client.shutdown(socket.SHUT_RDWR)
        self.client.close()

    def send_data_to_plc(self, device_type, device_no, data_format, data):
        command = f"WR {device_type}{device_no}{data_format} {data}\r"
        # print(command)
        try:
            self.client.send(command.encode("ascii"))
            # print("send : " + str(command.encode("ascii")))
            response = self.client.recv(
                32
            )  # 受信用バイト配列を定義しておく(responseのバイト数以上を設定しておく)
            # print(response)
            response = response.decode(
                "UTF-8"
            )  # PLCからの返答がbyteデータなのでUTF-8にデコード
            if response in "OK":
                # print("Ok")
                return response
            elif response in "E0":
                print(response)
                print("E0 Device No. Error")
            elif response in "E1":
                print(response)
                print("E1 Command Error")
            elif response in "E4":
                print(response)
                print("E4 Write Protected")
            # print(response)
            return response
        except Exception as ex:
            print(ex)
            return "Error"

    def read_data_from_plc(self, device_type, device_no, data_format, data_send):
        command = f"RDS {device_type}{device_no}{data_format} {data_send}\r"
        # print(command)
        try:
            self.client.send(command.encode("ascii"))
            # print("send : " + str(command.encode("ascii")))
            response = self.client.recv(
                32
            )  # 受信用バイト配列を定義しておく(responseのバイト数以上を設定しておく)
            # print(response)
            response = response.decode(
                "UTF-8"
            )  # PLCからの返答がbyteデータなのでUTF-8にデコード
            if response in "E0":
                print(response)
                print("E0 Device No. Error")
            elif response in "E1":
                print(response)
                print("E1 Command Error")
            # else:
            #     print("Response :", response)
            return response
        except Exception as ex:
            print(ex)
            return "Error"


if __name__ == "__main__":

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    conn = PLCConnect(client, "192.168.3.10", 8501)

    conn.connect()

    client.close()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    conn.connect()
