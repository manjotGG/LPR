import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path
import requests

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def download_yolo_model():
    """Download YOLOv8 pretrained model"""
    print("🤖 Downloading YOLOv8 model...")

    model_urls = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    }

    os.makedirs("models", exist_ok=True)

    for model_name, url in model_urls.items():
        model_path = f"models/{model_name}"
        if not os.path.exists(model_path):
            try:
                print(f"📥 Downloading {model_name}...")
                urllib.request.urlretrieve(url, model_path)
                print(f"✅ Downloaded: {model_path}")
            except Exception as e:
                print(f"❌ Error downloading {model_name}: {e}")
        else:
            print(f"✅ {model_name} already exists")

def download_license_plate_model():
    """Download specialized license plate detection model"""
    print("🎯 Downloading specialized license plate detection model...")

    # This would be a pre-trained model specifically for license plates
    model_urls = {
        "license_plate_yolov8.pt": "https://huggingface.co/keremberke/yolov5m-license-plate/resolve/main/best.pt"
    }

    os.makedirs("models", exist_ok=True)

    for model_name, url in model_urls.items():
        model_path = f"models/{model_name}"
        if not os.path.exists(model_path):
            try:
                print(f"📥 Downloading {model_name}...")
                response = requests.get(url)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                    print(f"✅ Downloaded: {model_path}")
                else:
                    print(f"❌ Failed to download {model_name} (Status: {response.status_code})")
            except Exception as e:
                print(f"❌ Error downloading {model_name}: {e}")
        else:
            print(f"✅ {model_name} already exists")

def create_sample_data():
    """Create sample data structure"""
    print("📁 Creating directory structure...")

    directories = [
        "models",
        "data/sample_images",
        "data/sample_videos", 
        "results/images",
        "results/videos",
        "datasets"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def download_sample_images():
    """Download sample Indian car images for testing"""
    print("🖼️ Downloading sample images...")

    sample_urls = [
        ("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", "sample_bus.jpg"),
    ]

    os.makedirs("data/sample_images", exist_ok=True)

    for url, filename in sample_urls:
        filepath = f"data/sample_images/{filename}"
        if not os.path.exists(filepath):
            try:
                print(f"📥 Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ Downloaded: {filepath}")
            except Exception as e:
                print(f"❌ Error downloading {filename}: {e}")
        else:
            print(f"✅ {filename} already exists")

def create_config_files():
    """Create configuration files"""
    print("⚙️ Creating configuration files...")

    # Create YOLOv8 training config for license plates
    yolo_config = """# YOLOv8 License Plate Detection Configuration

# Dataset paths
train: datasets/train/images
val: datasets/val/images  
test: datasets/test/images

# Number of classes
nc: 1

# Class names
names: 
  0: license_plate

# Training parameters
epochs: 100
batch_size: 16
img_size: 640

# Model architecture
model: yolov8n.pt
"""

    with open("config.yaml", "w") as f:
        f.write(yolo_config)
    print("✅ Created: config.yaml")

    # Create Indian license plate patterns file
    patterns_config = """# Indian License Plate Patterns
# This file contains patterns for Indian license plates

patterns:
  # Old format: XX ## XX ####
  old_format: "^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"

  # New format (BH series): XX ## XX ####  
  new_format: "^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"

  # Commercial vehicles: XX ## X ####
  commercial: "^[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}$"

# Common Indian state codes
state_codes:
  - DL  # Delhi
  - MH  # Maharashtra
  - KA  # Karnataka
  - TN  # Tamil Nadu
  - AP  # Andhra Pradesh
  - UP  # Uttar Pradesh
  - WB  # West Bengal
  - RJ  # Rajasthan
  - GJ  # Gujarat
  - HR  # Haryana
"""

    with open("indian_plates_config.yaml", "w") as f:
        f.write(patterns_config)
    print("✅ Created: indian_plates_config.yaml")

def main():
    print("🚗 Indian License Plate Recognition System - Setup")
    print("=" * 60)

    # Create directory structure
    create_sample_data()

    # Install requirements
    if not install_requirements():
        print("❌ Setup failed due to requirement installation issues")
        return

    # Download models
    download_yolo_model()
    download_license_plate_model()

    # Download sample data
    download_sample_images()

    # Create configuration files
    create_config_files()

    print("\n✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Run demo: python demo.py")
    print("2. Process an image: python indian_lpr_system.py --input data/sample_images/sample_bus.jpg")
    print("3. Process a video: python indian_lpr_system.py --input video.mp4 --video")

if __name__ == "__main__":
    main()