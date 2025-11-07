import Jetson.GPIO as GPIO

class GPIO_control:

    def __init__(self, pin_num=7, mode='output'):
        self.pin_num = pin_num
        try:
            GPIO.cleanup()
        except:
            pass  # 이미 정리되어 있거나 초기화되지 않은 경우
        
        try:
            GPIO.setmode(GPIO.BOARD)  # Jetson board numbering scheme
            if mode == 'input':
                GPIO.setup(self.pin_num, GPIO.IN)
            else:
                GPIO.setup(self.pin_num, GPIO.OUT, initial=GPIO.LOW)
            print("GPIO initialized successfully")
        except Exception as e:
            print(f"Error initializing GPIO: {e}")

    def send_gpio_pulse_start(self):
        """Start a GPIO pulse by setting pin HIGH"""
        try:
            GPIO.output(self.pin_num, GPIO.HIGH)
            print("GPIO pulse started (HIGH)")
        except Exception as e:
            print(f"Error starting GPIO pulse: {e}")

    def send_gpio_pulse_end(self):
        """End a GPIO pulse by setting pin LOW"""
        try:
            GPIO.output(self.pin_num, GPIO.LOW)
            print("GPIO pulse ended (LOW)")
        except Exception as e:
            print(f"Error ending GPIO pulse: {e}")

    def get_gpio_output_state(self):
        """Get current GPIO output pin state (0 or 1)"""
        try:
            return int(GPIO.input(self.pin_num))
        except:
            return 0

    def get_gpio_input_state(self):
        """Get current GPIO input pin state (0 or 1)"""
        try:
            return GPIO.input(self.pin_num)
        except:
            return 0

    def read_gpio_input(self):
        """Read GPIO input pin state (0 or 1)"""
        try:
            return GPIO.input(self.pin_num)
        except Exception as e:
            print(f"Error reading GPIO input: {e}")
            return 0

    def safe_gpio_cleanup(self):
        try:
            GPIO.cleanup()
            print("GPIO cleaned up successfully")
        except Exception as e:
            print(f"Error during GPIO cleanup: {e}")

if __name__ == "__main__":
    import time
    gpio_control = GPIO_control(pin_num=33, mode='input')
    try:
        print(f"Initial GPIO Input State: {gpio_control.get_gpio_input_state()}")
        while True:
            val = gpio_control.read_gpio_input()
            print(f"GPIO Input State: {val}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting and cleaning up GPIO.")
    finally:
        gpio_control.safe_gpio_cleanup()