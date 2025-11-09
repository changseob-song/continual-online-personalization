import time
import smbus, board
import ADS1x15
from Utils_Teleplot import Teleplot

class ADS1115_I2C:
    def __init__(self):
        JETSON_I2C_BUS = 7        # I2C bus of Orin to which multiplexer is attached.
        self.MUX_ADDRESS = 0x70        # I2C/SMBus address of multiplexer on exoskeleton.
        self.MUX_port = 2              # port on multiplexer to which ADC is attached.
        self.i2cbus = smbus.SMBus(JETSON_I2C_BUS)
        self.i2cbus.write_byte(self.MUX_ADDRESS, self.MUX_port) # Make sure to select correct MUX port
        
        self.ADS = ADS1x15.ADS1115(JETSON_I2C_BUS)
        self.ADS.setMode(0) # continuous mode
        self.ADS.setDataRate(self.ADS.DR_ADS111X_860) # Set fastest data rate

    def read_FSR(self, port):
        self.i2cbus.write_byte(self.MUX_ADDRESS, self.MUX_port) # Make sure to select correct MUX port
        self.ADS.requestADC(port)
        # time.sleep(0.0021) # Wait for conversion to complete (1/860s = ~1.16ms)
        fsr_value = self.ADS.getValue()

        return fsr_value

def main():
    teleplot = Teleplot()
    FSR_ADC = ADS1115_I2C()

    while True:
        # print("{:.3f}\t{:.3f}".format(*FSR_ADC.read_FSRs()[:2]))
        # print("{:.3f}\t{:.3f}".format(*FSR_ADC.read_FSRs()[2:]))

        fsr_log_start = time.time()
        L = FSR_ADC.read_FSR(0)
        R = FSR_ADC.read_FSR(1)
        fsr_log_time = time.time() - fsr_log_start

        teleplot.sendTelemetry('FSR1 value', L)
        teleplot.sendTelemetry('FSR2 value', R)
        teleplot.sendTelemetry('FSR log time', fsr_log_time)

        # time.sleep(0.01)
if __name__ == '__main__':
    main()