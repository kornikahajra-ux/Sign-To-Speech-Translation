import cv2
import numpy as np
import os
from pathlib import Path

def get_global_bbox(frame_paths, threshold=25):
    """Calculates a single bounding box that covers all motion in 40 frames."""
    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    accumulated_mask = np.zeros_like(first_frame)
    
    prev_frame = first_frame
    for i in range(1, len(frame_paths)):
        curr_frame = cv2.imread(str(frame_paths[i]), cv2.IMREAD_GRAYSCALE)
        
        curr_blur = cv2.GaussianBlur(curr_frame, (5, 5), 0)
        prev_blur = cv2.GaussianBlur(prev_frame, (5, 5), 0)
        
        
        diff = cv2.absdiff(curr_blur, prev_blur)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        accumulated_mask = cv2.bitwise_or(accumulated_mask, mask)
        prev_frame = curr_frame

    
    contours, _ = cv2.findContours(accumulated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None 
    
    x_coords, y_coords = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        x_coords.extend([x, x + w])
        y_coords.extend([y, y + h])
    
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)


input_root = r"C:\Users\Kornika\Desktop\Files\Society Tasks\MACS DTU\SignToSpeech\Training\Dataset"
output_root = r"C:\Users\Kornika\Desktop\Files\Society Tasks\MACS DTU\SignToSpeech\Training\PreprocessedFromScratch"

def process_dataset(input_root, output_root, target_size=(128, 128)):

    input_path = Path(input_root)
    output_path = Path(output_root)

    for label_folder in input_path.iterdir():
        if not label_folder.is_dir(): continue
        
        for sequence_folder in label_folder.iterdir():
            if not sequence_folder.is_dir(): continue
            
            
            frames = sorted(list(sequence_folder.glob('*.jpg'))) 
            if len(frames) == 0: continue

            
            bbox = get_global_bbox(frames)
            if bbox is None: continue
            
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            
            side = max(w, h)
            cx, cy = x1 + w//2, y1 + h//2
            
            
            nx1 = max(0, cx - side//2)
            ny1 = max(0, cy - side//2)
            nx2 = nx1 + side
            ny2 = ny1 + side

            
            seq_out_dir = output_path / label_folder.name / sequence_folder.name
            seq_out_dir.mkdir(parents=True, exist_ok=True)

            
            for f_path in frames:
                img = cv2.imread(str(f_path))
                
                crop = img[ny1:ny2, nx1:nx2]
                
                final_img = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
                
                cv2.imwrite(str(seq_out_dir / f_path.name), final_img)

    print("Preprocessing Complete!")

# Usage
process_dataset(input_root, output_root)


import cv2
import numpy as np
import os
from pathlib import Path

def process_temporal_mean(input_root, output_root, target_size=(128, 128)):
    input_path = Path(input_root)
    output_path = Path(output_root)

    for label_folder in input_path.iterdir():
        if not label_folder.is_dir(): continue
        
        for sequence_folder in label_folder.iterdir():
            if not sequence_folder.is_dir(): continue
            
            
            frame_files = sorted(list(sequence_folder.glob('*.jpg')))
            if len(frame_files) < 40: continue 

            frames_gray = []
            for f in frame_files:
                img = cv2.imread(str(f))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                frames_gray.append(blurred)

            
            stack = np.array(frames_gray, dtype=np.float32)
            mean_frame = np.mean(stack, axis=0).astype(np.uint8)

            
            seq_out_dir = output_path / label_folder.name / sequence_folder.name
            seq_out_dir.mkdir(parents=True, exist_ok=True)

            
            for i, frame in enumerate(frames_gray):
                
                diff = cv2.absdiff(frame, mean_frame)
                
                
                normalized_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
                
                
                _, final_mask = cv2.threshold(normalized_diff, 30, 255, cv2.THRESH_TOZERO)

                
                out_name = f"frame_{i:04d}.jpg"
                cv2.imwrite(str(seq_out_dir / out_name), final_mask)

    print("Temporal Mean Subtraction Preprocessing Complete!")

# Usage
input_root = r"C:\Users\Kornika\Desktop\Files\Society Tasks\MACS DTU\SignToSpeech\Training\PreprocessedFromScratch"
output_root = r"C:\Users\Kornika\Desktop\Files\Society Tasks\MACS DTU\SignToSpeech\Training\PreprocessedMotionData"
process_temporal_mean(input_root, output_root)