import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)  # Jetson board numbering scheme
mode = GPIO.getmode()
print(f"GPIO mode set to: {mode}")

# GPIO.setup(9, GPIO.IN)
# GPIO.setup(1, GPIO.IN)
GPIO.setup(29, GPIO.IN)

try:
    while True:

        # input_state = GPIO.input(9)
        # print(f"GPIO Input State on pin 9: {input_state}")

        # input_state = GPIO.input(1)
        # print(f"\nGPIO Input State on pin 29: {input_state}")

        input_state = GPIO.input(29)
        print(f"GPIO Input State on pin 29: {input_state}", end='\r')

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nExiting program.")

finally:
    GPIO.cleanup()
    print("GPIO cleanup complete.")
