import cv2
import numpy as np 
import tensorflow as tf 
import os 
import platform 
import collections
 
MODEL_PATH = 'Sign_Translation_Model_6.h5'
TARGET_SIZE = (128, 128) 
WINDOW_SIZE = 40 
SMOOTHING_WINDOW = 5 

CLASS_THRESHOLDS = {
            'COME': 0.90,
            'HELLO': 0.30,
            'HELP': 0.50,
            'NO': 0.45,
            'NOTHING': 0.35,
            'PLEASE': 0.80,
            'SORRY': 0.50,
            'STOP': 0.55,
            'THANK YOU': 0.45,
            'YES': 0.20
        }

model = tf.keras.models.load_model(MODEL_PATH) 
labels = ['COME', 'HELLO', 'HELP', 'NO', 'NOTHING', 'PLEASE', 'SORRY', 'STOP', 'THANK YOU', 'YES']

def speak_text(text):
    """Uses the system's native command line to speak.""" 
    current_os = platform.system() 
    if current_os == "Windows":
        
        cmd = f'PowerShell -Command "Add-Type -AssemblyName System.Speech; ' \
              f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"' 
        os.system(cmd) 
    elif current_os == "Darwin": # macOS 
        os.system(f'say "{text}"')
    else: # Linux 
        os.system(f'espeak "{text}"')

def preprocess_frame(frame, mean_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    diff = cv2.absdiff(blurred, mean_frame)
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    _, motion = cv2.threshold(normalized, 30, 255, cv2.THRESH_TOZERO)

    resized_motion = cv2.resize(motion, TARGET_SIZE)

    return resized_motion.astype('float32') / 255.0, blurred
    

cap = cv2.VideoCapture(0)
frame_buffer = collections.deque(maxlen=WINDOW_SIZE)
prediction_history = collections.deque(maxlen=SMOOTHING_WINDOW)
last_spoken = ""


ret, first_frame = cap.read()
gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
mean_frame = cv2.GaussianBlur(gray, (5, 5), 0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        processed, blurred = preprocess_frame(frame, mean_frame)
        alpha = 0.05
        mean_frame = cv2.addWeighted(
            blurred, alpha,
            mean_frame, 1 - alpha,
            0
        )

        motion_level = np.mean(processed)
        
        frame_buffer.append(processed)
        display_frame = frame.copy()

        

        if len(frame_buffer) == WINDOW_SIZE:
            
            input_data = np.expand_dims(np.array(frame_buffer), axis=(0, -1))
            
            preds = model.predict(input_data, verbose=0)[0]
            idx = np.argmax(preds)
            confidence = preds[idx]
            current_sign = labels[idx]
            
            prediction_history.append(current_sign)

            threshold = CLASS_THRESHOLDS.get(current_sign, 0.5)

            if confidence > threshold:

                
                if current_sign == "NOTHING":
                    if motion_level < 0.03 and prediction_history.count("NOTHING") >= 3:
                        display_label = "NOTHING"
                    else:
                        display_label = ""
    
                
                else:
                    if prediction_history.count(current_sign) >= 3 and current_sign != last_spoken:
                        speak_text(current_sign)
                        last_spoken = current_sign
                        display_label = current_sign

            
            cv2.putText(display_frame, f"Sign: {current_sign} ({confidence:.2f})", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            
            bar_x, bar_y = 10, 80
            bar_width = 300
            bar_height = 20

            filled_width = int(confidence * bar_width)

            
            cv2.rectangle(display_frame, 
                          (bar_x, bar_y), 
                          (bar_x + bar_width, bar_y + bar_height), 
                          (50, 50, 50), -1)

            
            cv2.rectangle(display_frame, 
                          (bar_x, bar_y), 
                          (bar_x + filled_width, bar_y + bar_height), 
                          (0, 255, 0), -1)

            

        cv2.imshow('Live Sign-to-Speech (Scratch-Trained)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    cap.release()
    cv2.destroyAllWindows()