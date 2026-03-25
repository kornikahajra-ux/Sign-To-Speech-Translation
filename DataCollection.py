import cv2
import os
import time
from IPython.display import display, Image, clear_output


SIGN_NAME = r"HELLO"  # Change this for each sign
DATA_PATH = r"C:\Users\Kornika\Desktop\Files\Society Tasks\MACS DTU\SignToSpeech\TrainingV2\DatasetExtra"
NUM_SAMPLES = 20     
FRAMES_PER_CLIP = 40 
DELAY_BETWEEN_CLIPS = 60


# Create base directory
if not os.path.exists(os.path.join(DATA_PATH, SIGN_NAME)):
    os.makedirs(os.path.join(DATA_PATH, SIGN_NAME))

cap = cv2.VideoCapture(0)


input("Position yourself in the camera and press Enter to start recording...")

try:
    for sample in range(1, NUM_SAMPLES + 1):
        # 1. THE COUNTDOWN PHASE:  
        for i in range(DELAY_BETWEEN_CLIPS, 0, -1):
            ret, frame = cap.read()
            if not ret: break
            
            
            cv2.putText(frame, f"GET READY: {i}s", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Adjust Lighting/Hand for Sample {sample}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            display(Image(data=buffer))
            clear_output(wait=True)
            time.sleep(1) 

        # 2. THE RECORDING PHASE 
        sample_path = os.path.join(DATA_PATH, SIGN_NAME, str(sample).zfill(2))
        os.makedirs(sample_path, exist_ok=True)
        
        for frame_num in range(FRAMES_PER_CLIP):
            ret, frame = cap.read()
            if not ret: break
            
            
            img_path = os.path.join(sample_path, f"frame_{frame_num:02d}.jpg")
            cv2.imwrite(img_path, frame)
            
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"RECORDING SAMPLE {sample}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', display_frame)
            display(Image(data=buffer))
            clear_output(wait=True)
            
            
            time.sleep(0.05) 
            
        print(f"Sample {sample} captured successfully!")

finally:
    cap.release()
    print("Camera turned off.")