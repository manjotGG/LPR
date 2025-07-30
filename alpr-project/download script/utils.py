import cv2
import numpy as np
import re
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

class IndianPlateValidator:
    """Validator for Indian license plate patterns"""

    def __init__(self, config_path="indian_plates_config.yaml"):
        """Initialize with configuration file"""
        self.patterns = {}
        self.state_codes = []

        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.patterns = config.get('patterns', {})
                self.state_codes = config.get('state_codes', [])

    def validate_plate_format(self, text):
        """
        Validate if text matches Indian license plate format

        Args:
            text (str): License plate text

        Returns:
            bool: True if valid format
        """
        if not text or len(text) < 8:
            return False

        # Clean text
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Check against patterns
        for pattern_name, pattern in self.patterns.items():
            if re.match(pattern, cleaned):
                return True

        return False

    def extract_state_code(self, text):
        """
        Extract state code from license plate

        Args:
            text (str): License plate text

        Returns:
            str: State code or empty string
        """
        if not text or len(text) < 2:
            return ""

        # Extract first two characters as potential state code
        state_code = text[:2].upper()

        if state_code in self.state_codes:
            return state_code

        return ""

class ImageProcessor:
    """Image processing utilities for license plate recognition"""

    @staticmethod
    def enhance_plate_image(image):
        """
        Enhance license plate image for better OCR

        Args:
            image: Input image (numpy array)

        Returns:
            numpy array: Enhanced image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply bilateral filter
        filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

        return processed

    @staticmethod
    def resize_maintain_aspect(image, target_width=None, target_height=None):
        """
        Resize image while maintaining aspect ratio

        Args:
            image: Input image
            target_width: Target width
            target_height: Target height

        Returns:
            numpy array: Resized image
        """
        height, width = image.shape[:2]

        if target_width and not target_height:
            # Calculate height based on width
            aspect_ratio = height / width
            target_height = int(target_width * aspect_ratio)
        elif target_height and not target_width:
            # Calculate width based on height
            aspect_ratio = width / height
            target_width = int(target_height * aspect_ratio)
        elif not target_width and not target_height:
            return image

        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

class TextCleaner:
    """Text processing utilities for license plate recognition"""

    @staticmethod
    def clean_ocr_text(text):
        """
        Clean OCR text for license plates

        Args:
            text (str): Raw OCR text

        Returns:
            str: Cleaned text
        """
        if not text:
            return ""

        # Convert to uppercase
        text = text.upper()

        # Replace common OCR mistakes
        replacements = {
            '0': 'O',  # Zero to O in some contexts
            'O': '0',  # O to zero in numeric contexts
            '1': 'I',  # One to I
            'I': '1',  # I to one
            '5': 'S',  # Five to S
            'S': '5',  # S to five
            '8': 'B',  # Eight to B
            'B': '8',  # B to eight
        }

        # Remove special characters except space
        cleaned = re.sub(r'[^A-Z0-9\s]', '', text)

        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Remove spaces for compact format
        compact = re.sub(r'\s', '', cleaned)

        return compact

    @staticmethod
    def format_indian_plate(text):
        """
        Format text as Indian license plate

        Args:
            text (str): Cleaned license plate text

        Returns:
            str: Formatted license plate
        """
        if not text or len(text) < 8:
            return text

        # Remove all spaces and special characters
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())

        if len(clean) == 10:
            # Format as: XX00XX0000
            return f"{clean[:2]} {clean[2:4]} {clean[4:6]} {clean[6:]}"
        elif len(clean) >= 8:
            # Handle variable length plates
            return clean

        return text

class Visualizer:
    """Visualization utilities"""

    @staticmethod
    def plot_detection_results(image, detections, save_path=None):
        """
        Plot detection results on image

        Args:
            image: Input image
            detections: List of detection results
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))

        # Convert BGR to RGB for matplotlib
        if len(image.shape) == 3:
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image

        plt.imshow(display_image)
        plt.title("License Plate Detection Results")
        plt.axis('off')

        # Add detection boxes and text
        for i, detection in enumerate(detections):
            bbox = detection.get('bbox', (0, 0, 0, 0))
            text = detection.get('text', 'Unknown')
            confidence = detection.get('detection_confidence', 0.0)

            x1, y1, x2, y2 = bbox

            # Draw rectangle
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'g-', linewidth=2)

            # Add text
            plt.text(x1, y1-10, f"{text} ({confidence:.2f})", 
                    color='green', fontsize=12, weight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved: {save_path}")

        plt.show()

    @staticmethod
    def create_comparison_plot(original, processed, title="Image Processing Comparison"):
        """
        Create before/after comparison plot

        Args:
            original: Original image
            processed: Processed image
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Original image
        if len(original.shape) == 3:
            ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            ax1.imshow(original, cmap='gray')
        ax1.set_title("Original")
        ax1.axis('off')

        # Processed image
        if len(processed.shape) == 3:
            ax2.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        else:
            ax2.imshow(processed, cmap='gray')
        ax2.set_title("Processed")
        ax2.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

def test_utilities():
    """Test utility functions"""
    print("🧪 Testing utility functions...")

    # Test plate validator
    validator = IndianPlateValidator()
    test_plates = ["DL12AB1234", "MH01BC5678", "INVALID", "KA03MN9993"]

    print("\n📋 Plate Validation Tests:")
    for plate in test_plates:
        is_valid = validator.validate_plate_format(plate)
        state = validator.extract_state_code(plate)
        print(f"  {plate}: Valid={is_valid}, State={state}")

    # Test text cleaner
    cleaner = TextCleaner()
    test_texts = ["DL 12 AB 1234", "K@03|1||9993", "MH-01-BC-5678"]

    print("\n🧹 Text Cleaning Tests:")
    for text in test_texts:
        cleaned = cleaner.clean_ocr_text(text)
        formatted = cleaner.format_indian_plate(cleaned)
        print(f"  '{text}' -> '{cleaned}' -> '{formatted}'")

    print("\n✅ Utility tests completed!")

if __name__ == "__main__":
    test_utilities()