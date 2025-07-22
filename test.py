import cv2
import numpy as np
import easyocr
import imutils
from matplotlib import pyplot as plt

def preprocess_for_ocr(image):
    """Enhanced preprocessing for better OCR results"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Remove noise and smooth
    denoised = cv2.medianBlur(gray, 3)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Try different thresholding methods
    # Method 1: Otsu's thresholding
    _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive thresholding
    thresh2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Method 3: Inverted thresholding (for dark text on light background)
    _, thresh3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return [thresh1, thresh2, thresh3, enhanced], ['Otsu', 'Adaptive', 'Inverted Otsu', 'Enhanced']



# === 1. Load Image ===
img = cv2.imread('test2.jpg')
if img is None:
    raise FileNotFoundError("Image not found. Please check the path.")

original_img = img.copy()
print(f"Image shape: {img.shape}")

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# === 2. Direct OCR Approach ===
print("Using direct OCR approach on full image...")

# Use the entire image for OCR
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
processed_images, method_names = preprocess_for_ocr(gray)

# Try OCR on the full image with different preprocessing
reader = easyocr.Reader(['en'], gpu=False)

# Store all detected texts with their bounding boxes and confidence
all_detections = []
license_plate_candidates = []

for i, (processed_img, method_name) in enumerate(zip(processed_images, method_names)):
    print(f"Trying OCR with {method_name} preprocessing on full image...")
    results = reader.readtext(processed_img)
    
    if results:
        for result in results:
            bbox, text, confidence = result
            print(f"Detected text ({method_name}): '{text}' (confidence: {confidence:.2f})")
            
            # Store all results
            all_detections.append((bbox, text, confidence, method_name))
            
            # Check if this looks like a license plate
            clean_text = ''.join(c for c in text if c.isalnum())
            if (confidence > 0.5 and 
                len(clean_text) >= 4 and 
                any(c.isdigit() for c in clean_text) and 
                any(c.isalpha() for c in clean_text)):
                license_plate_candidates.append((bbox, text, confidence, method_name))

# Find the best license plate candidate
best_plate = None
if license_plate_candidates:
    # Sort by confidence and select the best one
    license_plate_candidates.sort(key=lambda x: x[2], reverse=True)
    best_plate = license_plate_candidates[0]
    bbox, text, confidence, method_name = best_plate
    
    print(f"\nBest License Plate Candidate: '{text}' (confidence: {confidence:.2f}, method: {method_name})")
    
    # Create annotated image
    annotated_img = original_img.copy()
    
    # Convert bbox coordinates (EasyOCR returns normalized coordinates)
    bbox = np.array(bbox).astype(int)
    
    # Draw bounding box
    cv2.polylines(annotated_img, [bbox], True, (0, 255, 0), 3)
    
    # Add text annotation above the bounding box
    text_position = (bbox[0][0], bbox[0][1] - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_img, text, text_position, font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display and save result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected License Plate: {text} (Confidence: {confidence:.2f})')
    plt.axis('off')
    plt.show()
    
    # Save the annotated image
    cv2.imwrite('annotated_license_plate.jpg', annotated_img)
    print(f"Annotated image saved as 'annotated_license_plate.jpg'")
    
else:
    print("No license plate candidates found in the detected text.")
    
    # Still create an annotated image with all detected text
    if all_detections:
        annotated_img = original_img.copy()
        
        for bbox, text, confidence, method_name in all_detections:
            if confidence > 0.5:  # Only show high-confidence detections
                bbox = np.array(bbox).astype(int)
                
                # Different colors for different methods
                color_map = {
                    'Otsu': (0, 255, 0),
                    'Adaptive': (255, 0, 0), 
                    'Inverted Otsu': (0, 0, 255),
                    'Enhanced': (255, 255, 0)
                }
                color = color_map.get(method_name, (0, 255, 0))
                
                # Draw bounding box
                cv2.polylines(annotated_img, [bbox], True, color, 2)
                
                # Add text annotation
                text_position = (bbox[0][0], bbox[0][1] - 5)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(annotated_img, f'{text}({confidence:.2f})', text_position, 
                           font, 0.7, color, 2, cv2.LINE_AA)
        
        # Display and save
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.title('All Detected Text')
        plt.axis('off')
        plt.show()
        
        cv2.imwrite('annotated_all_text.jpg', annotated_img)
        print("Image with all detected text saved as 'annotated_all_text.jpg'")

print("Processing complete!")