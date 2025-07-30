
import cv2
import numpy as np
import easyocr
import torch
import re
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")

class ImprovedIndianLPR:
    def __init__(self, device='cpu'):
        self.device = device
        self.setup_models()
        self.setup_ocr()
        self.indian_patterns = self.load_indian_patterns()

    def setup_models(self):
        """Initialize YOLO model with error handling"""
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            self.model = None

    def setup_ocr(self):
        """Initialize EasyOCR with optimized settings"""
        try:
            self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
            print("✅ EasyOCR initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing EasyOCR: {e}")
            self.reader = None

    def load_indian_patterns(self):
        """Load Indian license plate patterns for validation"""
        patterns = [
            r'^[A-Z]{2}\s*\d{2}\s*[A-Z]{1,2}\s*\d{4}$',  # Standard format: MH 20 DV 2366
            r'^[A-Z]{2}\s*\d{2}\s*[A-Z]{1}\s*\d{4}$',    # Old format: DL 12 A 1234
            r'^[A-Z]{3}\s*\d{4}$',                        # Three letter format
            r'^[A-Z]{2}\s*\d{2}\s*[A-Z]{2}\s*\d{1,4}$',  # Variable digits
        ]
        return patterns

    def enhance_image_for_ocr(self, image):
        """Apply multiple preprocessing techniques to improve OCR accuracy"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced1 = clahe.apply(gray)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced1, (3, 3), 0)

        # Bilateral filter to preserve edges while reducing noise
        bilateral = cv2.bilateralFilter(blurred, 11, 17, 17)

        # Multiple thresholding approaches
        # Otsu's thresholding
        _, thresh1 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Adaptive thresholding
        thresh2 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        morph2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)

        # Return multiple processed versions for testing
        return {
            'original': gray,
            'clahe': enhanced1,
            'bilateral': bilateral,
            'otsu': morph1,
            'adaptive': morph2
        }

    def fix_character_confusion(self, text):
        """Fix common OCR character confusion issues for Indian license plates"""
        if not text:
            return text

        # Common character confusions in license plates
        corrections = {
            # Letters often confused with numbers
            'H': ['8', 'B'],
            'O': ['0', 'Q'],
            'I': ['1', 'l'],
            'S': ['5'],
            'Z': ['2'],
            'G': ['6'],
            'D': ['0'],
            'B': ['8', '6'],
            'R': ['8'],
            'A': ['4'],
            'E': ['3'],
            'T': ['7'],

            # Numbers often confused with letters  
            '0': ['O', 'D', 'Q'],
            '1': ['I', 'l'],
            '2': ['Z'],
            '3': ['E'],
            '4': ['A'],
            '5': ['S'],
            '6': ['G', 'B'],
            '7': ['T'],
            '8': ['B', 'H', 'R'],
            '9': ['g']
        }

        # Clean text first
        text = text.upper().strip()
        text = re.sub(r'[^A-Z0-9\s]', '', text)

        # Try to match Indian license plate patterns
        for pattern in self.indian_patterns:
            if re.match(pattern, text):
                return text

        # If no direct match, try character corrections
        # For Indian plates: typically 2 letters + 2 digits + 1-2 letters + 4 digits
        parts = text.split()
        if len(parts) >= 3:
            corrected_parts = []

            # First part should be 2 letters (state code)
            part1 = parts[0]
            if len(part1) == 2:
                corrected_part1 = ""
                for char in part1:
                    if char.isdigit():
                        # Convert digit to most likely letter
                        if char == '0': corrected_part1 += 'O'  # But actually should be letter
                        elif char == '1': corrected_part1 += 'I'
                        elif char == '2': corrected_part1 += 'Z'
                        elif char == '3': corrected_part1 += 'E'
                        elif char == '4': corrected_part1 += 'A'
                        elif char == '5': corrected_part1 += 'S'
                        elif char == '6': corrected_part1 += 'G'
                        elif char == '7': corrected_part1 += 'T'
                        elif char == '8': corrected_part1 += 'B'
                        else: corrected_part1 += char
                    else:
                        corrected_part1 += char
                corrected_parts.append(corrected_part1)
            else:
                corrected_parts.append(part1)

            # Second part should be 2 digits
            if len(parts) > 1:
                part2 = parts[1]
                corrected_part2 = ""
                for char in part2:
                    if char.isalpha():
                        # Convert letter to most likely digit
                        if char == 'O': corrected_part2 += '0'
                        elif char == 'I': corrected_part2 += '1'
                        elif char == 'Z': corrected_part2 += '2'
                        elif char == 'E': corrected_part2 += '3'
                        elif char == 'A': corrected_part2 += '4'
                        elif char == 'S': corrected_part2 += '5'
                        elif char == 'G': corrected_part2 += '6'
                        elif char == 'T': corrected_part2 += '7'
                        elif char == 'B': corrected_part2 += '8'
                        elif char == 'g': corrected_part2 += '9'
                        else: corrected_part2 += char
                    else:
                        corrected_part2 += char
                corrected_parts.append(corrected_part2)

            # Add remaining parts
            corrected_parts.extend(parts[2:])

            corrected_text = ' '.join(corrected_parts)

            # Check if corrected version matches pattern
            for pattern in self.indian_patterns:
                if re.match(pattern, corrected_text):
                    return corrected_text

        return text

    def extract_text_from_plate(self, plate_image):
        """Extract text using multiple OCR approaches"""
        if self.reader is None:
            return "", 0.0

        # Get multiple enhanced versions
        enhanced_images = self.enhance_image_for_ocr(plate_image)

        best_text = ""
        best_confidence = 0.0

        # Try OCR on different enhanced versions
        for version_name, enhanced_img in enhanced_images.items():
            try:
                # Resize image for better OCR (optimal size for license plates)
                height, width = enhanced_img.shape
                if height < 50:
                    scale_factor = 50 / height
                    new_width = int(width * scale_factor)
                    enhanced_img = cv2.resize(enhanced_img, (new_width, 50))
                elif height > 100:
                    scale_factor = 100 / height  
                    new_width = int(width * scale_factor)
                    enhanced_img = cv2.resize(enhanced_img, (new_width, 100))

                # Apply OCR
                results = self.reader.readtext(enhanced_img, detail=1, 
                                             allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ')

                for (bbox, text, confidence) in results:
                    if confidence > best_confidence and len(text.strip()) > 6:
                        best_text = text.strip()
                        best_confidence = confidence

            except Exception as e:
                print(f"OCR error on {version_name}: {e}")
                continue

        # Apply character confusion fixes
        if best_text:
            corrected_text = self.fix_character_confusion(best_text)
            return corrected_text, best_confidence

        return best_text, best_confidence

    def detect_license_plates(self, image):
        """Detect license plates using YOLO with fallback to contour detection"""
        detections = []

        if self.model is not None:
            try:
                # YOLO detection
                results = self.model(image, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = box.conf[0].item()
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'method': 'YOLO'
                            })
            except Exception as e:
                print(f"YOLO detection failed: {e}")

        # Fallback: Contour-based detection
        if not detections:
            detections = self.contour_based_detection(image)

        return detections

    def contour_based_detection(self, image):
        """Fallback contour-based license plate detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Check aspect ratio (license plates are typically wider than tall)
                aspect_ratio = w / h
                if 2.0 < aspect_ratio < 5.0:
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': 0.5,  # Default confidence for contour detection
                        'method': 'Contour'
                    })

        return detections

    def annotate_image(self, image, detections, texts):
        """Annotate image with bounding boxes and detected text"""
        annotated = image.copy()

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']

            x1, y1, x2, y2 = bbox

            # Draw bounding box
            color = (0, 255, 0) if method == 'YOLO' else (255, 0, 0)  # Green for YOLO, Blue for Contour
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Get detected text for this detection
            detected_text = texts[i] if i < len(texts) else "No text detected"
            text_confidence = texts[i + len(detections)] if i + len(detections) < len(texts) else 0

            # Create label with detected text
            label = f"{detected_text}"

            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw text background
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), 
                         (x1 + text_width + 10, y1), color, -1)

            # Draw text
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)

            # Draw confidence and method info
            info_label = f"{method}: {confidence:.2f}"
            cv2.putText(annotated, info_label, (x1, y2 + 20), 
                       font, 0.4, color, 1)

        return annotated

    def process_image(self, image_path, output_dir=None):
        """Main processing function"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"🔍 Processing: {image_path}")

        # Detect license plates
        detections = self.detect_license_plates(image)

        if not detections:
            print("❌ No license plates detected")
            return None

        # Extract text from each detected plate
        detected_texts = []
        confidences = []

        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Crop plate region with some padding
            padding = 10
            y1_pad = max(0, y1 - padding)
            y2_pad = min(image.shape[0], y2 + padding)
            x1_pad = max(0, x1 - padding)
            x2_pad = min(image.shape[1], x2 + padding)

            plate_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

            if plate_crop.size > 0:
                # Extract text
                text, confidence = self.extract_text_from_plate(plate_crop)
                detected_texts.append(text)
                confidences.append(confidence)

                print(f"✅ Detected: {text} (Confidence: {confidence:.2f})")
            else:
                detected_texts.append("Invalid crop")
                confidences.append(0.0)

        # Annotate image
        annotated_image = self.annotate_image(image, detections, detected_texts + confidences)

        # Save annotated image
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = Path(image_path).stem
            output_path = os.path.join(output_dir, f"annotated_{filename}.jpg")
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Annotated image saved: {output_path}")

        return {
            'image': annotated_image,
            'detections': detections,
            'texts': detected_texts,
            'confidences': confidences
        }

def main():
    parser = argparse.ArgumentParser(description='Improved Indian License Plate Recognition')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', default='results', help='Output directory')
    args = parser.parse_args()

    # Initialize system
    lpr = ImprovedIndianLPR()

    # Process image
    try:
        result = lpr.process_image(args.input, args.output)
        if result:
            print(f"\n🎯 Processing complete!")
            print(f"   Detections: {len(result['detections'])}")
            for i, text in enumerate(result['texts']):
                conf = result['confidences'][i]
                print(f"   Plate {i+1}: {text} (Confidence: {conf:.2f})")
        else:
            print("❌ No results")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
