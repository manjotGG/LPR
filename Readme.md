# YOLOv5 License Plate Recognition

This project demonstrates how to detect and read license plates from images using a custom-trained YOLOv5 model and EasyOCR. The script combines object detection (YOLOv5) to localize license plates and optical character recognition (EasyOCR) to extract the plate numbers.

## Features

- Detects license plates in images using a custom YOLOv5 model.
- Crops and enhances detected plate regions for improved OCR accuracy.
- Uses EasyOCR for robust license plate text recognition.
- Visualizes and saves results with bounding boxes and recognized text.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [matplotlib](https://matplotlib.org/)
- YOLOv5 weights file (e.g., `xyz.pt`, trained for license plate detection)
- Test image (e.g., `test2.jpg`)

Install required Python packages:
```bash
pip install torch torchvision
pip install opencv-python
pip install easyocr
pip install matplotlib
```

## Usage

1. **Place your YOLOv5 weights file** (`xyz.pt`) in the working directory.
2. **Place your test image** (`test2.jpg`) in the working directory.
3. **Run the script:**
    ```bash
    python yolo.py
    ```
4. **Outputs:**
   - Annotated image saved as `output_annotated.jpg`.
   - Cropped license plate images saved as `cropped_plate_0.jpg`, `cropped_plate_1.jpg`, etc.
   - Detected license plate text printed to the console.

## Code Overview

- Loads a custom YOLOv5 model for license plate detection.
- Reads and processes the input image.
- Runs YOLOv5 to detect license plates.
- For each detected plate:
  - Crops the plate region.
  - Enhances it using CLAHE (Contrast Limited Adaptive Histogram Equalization).
  - Runs OCR with EasyOCR.
  - Draws bounding boxes and recognized text on the image.
  - Saves the cropped plate images.
- Displays and saves the final annotated image.

## Notes

- The model confidence threshold can be adjusted via `model.conf`.
- Make sure your YOLOv5 weights are trained specifically for license plate detection.
- For best OCR results, ensure the input images are clear and the plates are readable.

## References

- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)

## License

This project is for educational purposes. For commercial use, ensure you comply with the licenses of YOLOv5, EasyOCR, and OpenCV.