import serial
import numpy as np
import time
import os

print("="*60)
print("DATA COLLECTION STARTING...")
print("="*60)

# Configuration
ARDUINO_PORT = 'COM3'
SAMPLE_DURATION = 2.0
SAMPLE_RATE = 50
SAMPLES_PER_TRIAL = 100  # 2 seconds * 50 Hz

def collect_single_trial(ser, position, trial_num):
    """Collect one hit recording"""
    print(f"\n{'='*50}")
    print(f"Position {position}, Trial {trial_num + 1}/10")
    print(f"{'='*50}")
    
    input("Press ENTER when ready, then hit in 3 seconds...")
    
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("HIT NOW! Recording...")
    
    # Flush old data
    ser.reset_input_buffer()
    
    # Collect samples
    data = []
    start_time = time.time()
    timeout = SAMPLE_DURATION + 2
    
    while len(data) < SAMPLES_PER_TRIAL:
        if time.time() - start_time > timeout:
            print("⚠️  Timeout - retry")
            return None
            
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                value = int(line)
                data.append(value)
                
                # Show progress
                if len(data) % 25 == 0:
                    print(f"  Progress: {len(data)}/{SAMPLES_PER_TRIAL}")
        except:
            continue
    
    signal = np.array(data)
    max_val = np.max(signal)
    mean_val = np.mean(signal)
    
    print(f"✓ Recorded! Max={max_val}, Mean={mean_val:.1f}")
    
    # Check if signal looks good
    if max_val < 300:
        print("⚠️  Signal seems weak")
        retry = input("Accept anyway? (y/n): ")
        if retry.lower() != 'y':
            return None
    
    return signal

def main():
    print("\nSetting up...")
    
    # Create directory
    os.makedirs('data/raw', exist_ok=True)
    print("✓ Directory ready")
    
    # Connect to Arduino
    print(f"\nConnecting to {ARDUINO_PORT}...")
    try:
        ser = serial.Serial(ARDUINO_PORT, 115200, timeout=1)
        print("✓ Connected!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure Arduino IDE is closed!")
        return
    
    # Wait for reset
    time.sleep(2)
    ser.reset_input_buffer()
    
    print("\n" + "="*60)
    print("READY TO COLLECT DATA")
    print("="*60)
    print("Total: 50 trials (5 positions × 10 each)")
    print("Make sure you've marked 5 positions on your ruler!")
    print("="*60)
    
    input("\nPress ENTER to start...")
    
    # Main collection loop
    total_collected = 0
    
    for position in range(5):
        print("\n" + "*"*60)
        print(f"POSITION {position}")
        print(f"Hit the ruler at position {position} mark (mark {position})")
        print("*"*60)
        
        trials_done = 0
        
        while trials_done < 10:
            signal = collect_single_trial(ser, position, trials_done)
            
            if signal is None:
                print("Retrying this trial...")
                continue
            
            # Save
            filename = f'data/raw/position_{position}_trial_{trials_done}.npy'
            np.save(filename, signal)
            print(f"✓ Saved: {filename}")
            
            trials_done += 1
            total_collected += 1
            
            print(f"Progress: {total_collected}/50 total trials complete")
            
            time.sleep(0.5)
    
    ser.close()
    
    print("\n" + "="*60)
    print("✓✓✓ DATA COLLECTION COMPLETE! ✓✓✓")
    print("="*60)
    print(f"Total files: {total_collected}")
    print("Location: data/raw/")
    print("\nNext step:")
    print("  python src/extract_features.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
