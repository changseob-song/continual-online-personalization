import time
import socket
import os

class Teleplot:
    def __init__(self, host='127.0.0.1', port=47269):
        os.system('echo nc -u -w0 127.0.0.1 47269')
        self.teleplotAddr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def sendTelemetry(self, name, value):
        now = time.time() * 1000
        msg = f"{name}:{now}:{value}|g"
        self.sock.sendto(msg.encode(), self.teleplotAddr)

    def sendBatchTelemetry(self, data_dict):
        now = time.time() * 1000
        try:
            for name, value in data_dict.items():
                msg = f"{name}:{now}:{value}|g"
                self.sock.sendto(msg.encode(), self.teleplotAddr)
            return True  # Successfully sent
        except Exception as e:
            print(f"Error in sendBatchTelemetry: {e}")
            return False
