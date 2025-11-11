import can
import Utils_tmotor_v3, Utils_actuator_group
from Utils_ICM20948_I2C_pcb2 import ICM20948_I2C_IMUs
from Utils_ADS1115_I2C import ADS1115_I2C

class Exo:
    def __init__(self,):

        self.CAN_id_L = 1 # NEED TO FIRST IDENTIFY THIS using display_motor_data.py
        self.CAN_id_R = 2 # NEED TO FIRST IDENTIFY THIS using display_motor_data.py
        self.mtr_type = "AK80-9"
        self.control_freq_Hz = 100
        self.frame_length = 100  # Window size (in frames)
        self.frame_length_task = 200  # Window size for task estimator (in frames)

        # biotorque parameters
        self.scale_factor = 0.20
        self.delay_factor = 0  # Number of frames to delay the torque command
        self.max_torque = 10.0  # Maximum allowable torque (Nm)

        # note: motors zero themselves when actuation.Motors() runs
        _ = input("Press Enter to initialize motors: ")
        init_dict = {mtr_id: self.mtr_type for mtr_id in [self.CAN_id_L, self.CAN_id_R]}
        self.mtr_comms = Utils_actuator_group.ActuatorGroup([Utils_tmotor_v3.TMotorV3(mtr_id, self.mtr_type) for mtr_id in init_dict.keys()])

        # IMU initialization
        self.imus = ICM20948_I2C_IMUs()  # Back, Left hip, Right hip

        # FSR initialization
        self.fsr_adc = ADS1115_I2C()
        self.fsr_threshold = 15000  # Threshold value to detect foot contact

        # Specify the CAN interface and channel
        try:
            self.bus = can.Bus(interface='socketcan', channel='can0')  # Replace 'socketcan' and 'can0' with your actual interface and channel
            # print("CAN bus initialized successfully.")
        except Exception as e:
            print(f"Error initializing CAN bus: {e}")
        self.notifier = can.Notifier(self.bus, [])

    def update_readings(self, CAN_id):
        mtr_pos = self.mtr_comms.get_position(CAN_id, degrees=True)
        mtr_vel = self.mtr_comms.get_velocity(CAN_id, degrees=False)

        return mtr_pos, mtr_vel