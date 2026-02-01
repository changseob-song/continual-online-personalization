#wait untill jetson sends first data, working 
import socket
import json
import time
import csv
import os
import threading
import Utils_Teleplot

class Mocap_trigger:
    def __init__(self, server_ip, port_number):
        self.server_ip = server_ip
        self.port_number = port_number
        self.client = None

        self.GRF_R = 0.0
        self.GRF_L = 0.0
        self.time_sent = 0.0

        # Trigger handling
        self.trigger_received = False
        self.trigger_value = None

        # Logging control
        self.start_logging_received = False

        self.lock = threading.Lock()
        self.running = False

        self.first_data_received = threading.Event()
        self.start_logging_event = threading.Event()
        
        # Vicon timestamp tracking for precise synchronization
        self.latest_vicon_timestamp = None
        self.first_vicon_timestamp = None

        self.teleplot = Utils_Teleplot.Teleplot()

    def start_client(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print("[CONNECTING] Connecting to server...")
            self.client.connect((self.server_ip, self.port_number))
            print(f"[CONNECTED] Connected to server at {self.server_ip}:{self.port_number}")
        except ConnectionRefusedError:
            print(f"[ERROR] Cannot connect to server at {self.server_ip}:{self.port_number}")
            return

    def get_GRF(self):
        """Return the latest Ground Reaction Force data safely"""
        # Wait until at least one packet has arrived to ensure data validity
        self.first_data_received.wait()
        with self.lock:
            return self.GRF_L, self.GRF_R

    def stream_start(self):
        """Start the data receiving loop in a background thread"""
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
        print("[INFO] Mocap data stream thread started.")

    def _receive_loop(self):
        """Internal loop to continuously receive data"""
        buffer = ""
        while self.running:
            try:
                chunk = self.client.recv(4096).decode('utf-8')
                if not chunk:
                    print("[DISCONNECTED] Server closed the connection.")
                    self.running = False
                    break
                
                buffer += chunk

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
                    
                    # Check for start logging command
                    if line == "START_LOGGING":
                        with self.lock:
                            self.start_logging_received = True
                        self.start_logging_event.set()
                        print("[INFO] Start logging signal received from server!")
                        continue
                    
                    # Otherwise, try to parse as JSON data
                    try:
                        data = json.loads(line)
                        recv_time = time.time()
                        send_time = float(data.get("vicon_timestamp", 0.00))

                        with self.lock:
                            self.GRF_R = float(data.get("GRF_R", 0.000))
                            self.GRF_L = float(data.get("GRF_L", 0.000))
                            self.time_sent = send_time
                            
                            # Track Vicon timestamps for precise synchronization
                            self.latest_vicon_timestamp = send_time
                            if self.first_vicon_timestamp is None:
                                self.first_vicon_timestamp = send_time

                        latency = (recv_time - send_time) * 1000

                        if not self.first_data_received.is_set():
                            print("[INFO] First data received. Proceeding...")
                            self.first_data_received.set()


                    except json.JSONDecodeError:
                        print(f"[WARNING] JSON Decode Error on line: {line[:50]}...")
            
            except Exception as e:
                print(f"[ERROR] streaming loop failed: {e}")
                self.running = False
                break

        return self.GRF_L, self.GRF_R

    def get_gait_phase_percentages(self):
        # Optional: Wait until data is ready before proceeding
        self.first_data_received.wait()
        with self.lock:
            return self.gcp_r, self.gcp_l

    def check_trigger(self):
        """Check if a trigger was received and return its value. Returns None if no trigger."""
        with self.lock:
            if self.trigger_received:
                return self.trigger_value
            return None

    def wait_for_start_logging(self):
        """Wait for the server to send the start logging signal"""
        print("[INFO] Waiting for start logging signal from server...")
        self.start_logging_event.wait()
        print("[INFO] Start logging signal received.")
        return True

    def check_start_logging(self):
        """Check if start logging signal was received"""
        with self.lock:
            return self.start_logging_received

    def get_vicon_timestamps(self):
        """Get the latest and first Vicon timestamps for synchronization"""
        with self.lock:
            return self.latest_vicon_timestamp, self.first_vicon_timestamp

    def stop_streaming(self):
        self.running = False
        try:
            self.client.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        self.client.close()

if __name__ == "__main__":
    mocap_trigger = Mocap_trigger(server_ip="172.24.44.177", port_number=11)
    mocap_trigger.start_client()
    while True:
        mocap_trigger.stream_data()
        time.sleep(0.01)