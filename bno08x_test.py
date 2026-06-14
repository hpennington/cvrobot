import time
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_ROTATION_VECTOR,
)

def main():
    print("Initializing I2C bus...")
    # Blinka automatically maps board.SCL and board.SDA to the Jetson's active hardware I2C bus
    i2c = busio.I2C(board.SCL, board.SDA)

    print("Connecting to BNO08x sensor...")
    try:
        bno = BNO08X_I2C(i2c, address=0x4B)
    except ValueError as e:
        print(f"Error: Could not find BNO08x sensor. Check your wiring! {e}")
        return

    print("Sensor found! Enabling data streams...")
    # The BNO08x requires you to explicitly turn on the features you want to track
    # 100000 microseconds = 100ms interval (10Hz refresh rate)
    bno.enable_feature(BNO_REPORT_ACCELEROMETER, 100000)
    bno.enable_feature(BNO_REPORT_GYROSCOPE, 100000)
    bno.enable_feature(BNO_REPORT_ROTATION_VECTOR, 100000)

    print("\nReading data. Press Ctrl+C to stop.\n")
    try:
        while True:
            # 1. Read Linear Acceleration (m/s^2)
            accel_x, accel_y, accel_z = bno.acceleration
            print(f"Accel  -> X: {accel_x:7.2f} | Y: {accel_y:7.2f} | Z: {accel_z:7.2f} m/s^2")

            # 2. Read Gyroscope (rad/s)
            gyro_x, gyro_y, gyro_z = bno.gyro
            print(f"Gyro   -> X: {gyro_x:7.2f} | Y: {gyro_y:7.2f} | Z: {gyro_z:7.2f} rad/s")

            # 3. Read Rotation Vector (Quaternion orientation representation)
            quat_i, quat_j, quat_k, quat_real = bno.quaternion
            print(f"Quat   -> I: {quat_i:7.4f} | J: {quat_j:7.4f} | K: {quat_k:7.4f} | Real: {quat_real:7.4f}")
            print("-" * 65)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nScript stopped by user.")

if __name__ == "__main__":
    main()

