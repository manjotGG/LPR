import torch
import cv2
import easyocr
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

# === Load YOLOv5 License Plate Detector ===
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Replace with your trained weights
model.conf = 0.4  # Confidence threshold

# === Load Image ===
img_path = 'test2.jpg'
img = cv2.imread(img_path)
assert img is not None, "Image not found!"

# === Run YOLO Detection ===
results = model(img)

# Extract detections
detections = results.xyxy[0].cpu().numpy()

if len(detections) == 0:
    print("No license plate detected.")
else:
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        print(f"License Plate Detected at [{x1}, {y1}, {x2}, {y2}] with confidence {conf/255:.2f}")
        
        # Crop the plate region
        cropped_plate = img[y1:y2, x1:x2]

        # === Optional Preprocessing (CLAHE) ===
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # === OCR using EasyOCR ===
        reader = easyocr.Reader(['en'])
        results = reader.readtext(enhanced)

        text = "Not detected"
        if results:
            text = results[0][-2]
            print(f"Detected Text: {text}")

        # === Draw Result ===
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save or display cropped plate
        cv2.imwrite(f"cropped_plate_{i}.jpg", cropped_plate)

# === Show Final Annotated Image ===
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Final Result')
plt.axis('off')
plt.show()

# Save output
cv2.imwrite('output_annotated.jpg', img)
