import serial
import time

print("Testing Arduino connection on COM3...")

try:
    ser = serial.Serial('COM3', 115200, timeout=2)
    print("✓ Connected to COM3!")
    
    print("Waiting for Arduino reset...")
    time.sleep(2)
    
    print("\nReading 10 samples:")
    for i in range(10):
        line = ser.readline().decode('utf-8').strip()
        if line:
            print(f"  Sample {i+1}: {line}")
    
    ser.close()
    print("\n✓ SUCCESS! Arduino is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nQuick fixes:")
    print("  1. Close Arduino IDE completely")
    print("  2. Unplug Arduino, wait 3 seconds, plug back in")
    print("  3. Run this script again")
