import time

import board
from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15

# Create the I2C bus
i2c = board.I2C()

# Create the ADC object using the I2C bus
ads = ADS1115(i2c)

# # Create single-ended input on channel 0
# chan = AnalogIn(ads, ads1x15.Pin.A0)

# # Create differential input between channel 0 and 1
# # chan = AnalogIn(ads, ads1x15.Pin.A0, ads1x15.Pin.A1)

# print("{:>5}\t{:>5}".format("raw", "v"))

# while True:
#     print("{:>5}\t{:>5.3f}".format(chan.value, chan.voltage))
#     time.sleep(0.5)

# GPIO.setmode(GPIO.BOARD)  # Jetson board numbering scheme
# mode = GPIO.getmode()
# print(f"GPIO mode set to: {mode}")

# # GPIO.setup(9, GPIO.IN)
# # GPIO.setup(1, GPIO.IN)
# GPIO.setup(29, GPIO.IN)

# try:
#     while True:

#         # input_state = GPIO.input(9)
#         # print(f"GPIO Input State on pin 9: {input_state}")

#         # input_state = GPIO.input(1)
#         # print(f"\nGPIO Input State on pin 29: {input_state}")

#         input_state = GPIO.input(29)
#         print(f"GPIO Input State on pin 29: {input_state}", end='\r')

#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("\nExiting program.")

# finally:
#     GPIO.cleanup()
#     print("GPIO cleanup complete.")
