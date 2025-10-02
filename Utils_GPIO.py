import Jetson.GPIO as GPIO

class GPIO_control:

    def __init__(self, output_pin=7):
        self.output_pin = output_pin
        try:
            GPIO.cleanup()
        except:
            pass  # 이미 정리되어 있거나 초기화되지 않은 경우
        
        try:
            GPIO.setmode(GPIO.BOARD)  # Jetson board numbering scheme
            GPIO.setup(self.output_pin, GPIO.OUT, initial=GPIO.LOW)
            print("GPIO initialized successfully")
        except Exception as e:
            print(f"Error initializing GPIO: {e}")

    def send_gpio_pulse_start(self):
        """Start a GPIO pulse by setting pin HIGH"""
        try:
            GPIO.output(self.output_pin, GPIO.HIGH)
            print("GPIO pulse started (HIGH)")
        except Exception as e:
            print(f"Error starting GPIO pulse: {e}")

    def send_gpio_pulse_end(self):
        """End a GPIO pulse by setting pin LOW"""
        try:
            GPIO.output(self.output_pin, GPIO.LOW)
            print("GPIO pulse ended (LOW)")
        except Exception as e:
            print(f"Error ending GPIO pulse: {e}")

    def get_gpio_output_state(self):
        """Get current GPIO output pin state (0 or 1)"""
        try:
            return int(GPIO.input(self.output_pin))
        except:
            return 0

    def safe_gpio_cleanup(self):
        try:
            GPIO.cleanup()
            print("GPIO cleaned up successfully")
        except Exception as e:
            print(f"Error during GPIO cleanup: {e}")