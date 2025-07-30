import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import os

def download_sample_image():
    """Download a sample Indian car image for testing"""
    # Sample image URL (Indian car with license plate)
    sample_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"

    try:
        response = requests.get(sample_url)
        if response.status_code == 200:
            with open("sample_car.jpg", "wb") as f:
                f.write(response.content)
            print("✅ Sample image downloaded: sample_car.jpg")
            return "sample_car.jpg"
    except Exception as e:
        print(f"❌ Error downloading sample image: {e}")

    return None

def simple_ocr_demo(image_path):
    """
    Simple OCR demo using OpenCV edge detection and EasyOCR
    This is a fallback method when YOLO model is not available
    """
    print(f"🔍 Processing image: {image_path}")

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)

    # Edge detection
    edged = cv2.Canny(filtered, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Initialize EasyOCR
    print("🤖 Initializing EasyOCR...")
    reader = easyocr.Reader(['en'])

    plate_found = False

    # Look for rectangular contours (potential license plates)
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If contour has 4 corners, it might be a license plate
        if len(approx) == 4:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # Check if dimensions seem reasonable for a license plate
            aspect_ratio = w / h
            if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                # Draw rectangle on original image
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the license plate region
                plate_crop = gray[y:y+h, x:x+w]

                # Apply additional preprocessing for OCR
                plate_crop = cv2.bilateralFilter(plate_crop, 11, 17, 17)

                # Use EasyOCR to extract text
                try:
                    results = reader.readtext(plate_crop)

                    if results:
                        # Get the best result
                        best_result = max(results, key=lambda x: x[2])
                        text = best_result[1]
                        confidence = best_result[2]

                        if confidence > 0.3:  # Minimum confidence threshold
                            # Clean up the text
                            import re
                            cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())

                            # Add text to image
                            cv2.putText(img, f"{cleaned_text} ({confidence:.2f})", 
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            print(f"✅ Detected License Plate: {cleaned_text}")
                            print(f"📊 Confidence: {confidence:.2f}")
                            plate_found = True
                            break

                except Exception as e:
                    print(f"OCR Error: {e}")
                    continue

    if not plate_found:
        print("❌ No license plate detected")

    # Save result
    output_path = f"result_{Path(image_path).name}"
    cv2.imwrite(output_path, img)
    print(f"💾 Result saved: {output_path}")

    # Display using matplotlib (if running in notebook/interactive environment)
    try:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("License Plate Detection Result")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"display_{Path(image_path).name}", dpi=150, bbox_inches='tight')
        print(f"📊 Display image saved: display_{Path(image_path).name}")
    except Exception as e:
        print(f"Display warning: {e}")

def main():
    print("🚗 Indian License Plate Recognition - Simple Demo")
    print("=" * 50)

    # Check if sample image exists, if not download one
    if not os.path.exists("sample_car.jpg"):
        print("📥 Downloading sample image...")
        download_sample_image()

    # Test with sample image
    if os.path.exists("sample_car.jpg"):
        simple_ocr_demo("sample_car.jpg")
    else:
        print("❌ No sample image available. Please provide an image manually.")
        print("Usage: python demo.py <image_path>")

    print("\n✅ Demo completed!")
    print("\nTo use the full system with YOLO:")
    print("python indian_lpr_system.py --input sample_car.jpg --output results/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        simple_ocr_demo(sys.argv[1])
    else:
        main()