import time
import socket
import os

class Teleplot:
    def __init__(self, host='127.0.0.1', port=47269):
        os.system('echo nc -u -w0 127.0.0.1 47269')
        self.teleplotAddr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @staticmethod
    def _key(name):
        # "pos_L" -> "pos_L,pos": Teleplot puts series with the same label after the
        # comma into one chart, so each _L/_R pair shares a plot (legend keeps _L/_R).
        if name.endswith(('_L', '_R')):
            return f"{name},{name[:-2]}"
        return name

    def sendTelemetry(self, name, value):
        now = time.time() * 1000
        msg = f"{self._key(name)}:{now}:{value}|g"
        self.sock.sendto(msg.encode(), self.teleplotAddr)

    def sendBatchTelemetry(self, data_dict):
        now = time.time() * 1000
        try:
            for name, value in data_dict.items():
                msg = f"{self._key(name)}:{now}:{value}|g"
                self.sock.sendto(msg.encode(), self.teleplotAddr)
            return True  # Successfully sent
        except Exception as e:
            print(f"Error in sendBatchTelemetry: {e}")
            return False
