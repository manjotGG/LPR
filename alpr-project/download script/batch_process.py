import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import time
from indian_lpr_system import IndianLPRSystem
from utils import Visualizer
import glob

class BatchProcessor:
    def __init__(self, model_path='yolov8n.pt', confidence=0.25):
        """Initialize batch processor"""
        self.anpr = IndianLPRSystem(model_path, confidence)
        self.results = []

    def process_images_in_directory(self, input_dir, output_dir, extensions=['jpg', 'jpeg', 'png', 'bmp']):
        """
        Process all images in a directory

        Args:
            input_dir (str): Input directory path
            output_dir (str): Output directory path
            extensions (list): Supported image extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_files = []
        for ext in extensions:
            pattern = f"*.{ext}"
            image_files.extend(input_path.glob(pattern))
            image_files.extend(input_path.glob(pattern.upper()))

        if not image_files:
            print(f"❌ No image files found in: {input_dir}")
            return

        print(f"🔍 Found {len(image_files)} images to process")
        print(f"📤 Output directory: {output_dir}")

        # Process each image
        start_time = time.time()
        successful = 0
        total_detections = 0

        for i, image_file in enumerate(image_files, 1):
            print(f"\n📸 Processing {i}/{len(image_files)}: {image_file.name}")

            try:
                # Process image
                result = self.anpr.process_image(
                    str(image_file), 
                    save_result=True, 
                    output_dir=str(output_path)
                )

                if result and result['detections']:
                    successful += 1
                    detections_count = len(result['detections'])
                    total_detections += detections_count

                    # Store results
                    for detection in result['detections']:
                        self.results.append({
                            'filename': image_file.name,
                            'filepath': str(image_file),
                            'text': detection['text'],
                            'detection_confidence': detection['detection_confidence'],
                            'ocr_confidence': detection['ocr_confidence'],
                            'bbox_x1': detection['bbox'][0],
                            'bbox_y1': detection['bbox'][1],
                            'bbox_x2': detection['bbox'][2],
                            'bbox_y2': detection['bbox'][3]
                        })

                    print(f"✅ Detected {detections_count} license plate(s)")
                else:
                    print("❌ No license plates detected")

            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")

        # Calculate statistics
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / len(image_files)

        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total images: {len(image_files)}")
        print(f"   Successfully processed: {successful}")
        print(f"   Total detections: {total_detections}")
        print(f"   Processing time: {elapsed_time:.2f} seconds")
        print(f"   Average time per image: {avg_time:.2f} seconds")
        print(f"   Success rate: {(successful/len(image_files)*100):.1f}%")

        return successful, total_detections

    def process_videos_in_directory(self, input_dir, output_dir, extensions=['mp4', 'avi', 'mov', 'mkv']):
        """
        Process all videos in a directory

        Args:
            input_dir (str): Input directory path
            output_dir (str): Output directory path
            extensions (list): Supported video extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all video files
        video_files = []
        for ext in extensions:
            pattern = f"*.{ext}"
            video_files.extend(input_path.glob(pattern))
            video_files.extend(input_path.glob(pattern.upper()))

        if not video_files:
            print(f"❌ No video files found in: {input_dir}")
            return

        print(f"🎥 Found {len(video_files)} videos to process")

        # Process each video
        for i, video_file in enumerate(video_files, 1):
            print(f"\n🎬 Processing {i}/{len(video_files)}: {video_file.name}")

            output_video_path = output_path / f"processed_{video_file.name}"

            try:
                detections = self.anpr.process_video(
                    str(video_file),
                    str(output_video_path)
                )

                print(f"✅ Processed {video_file.name}: {len(detections)} detections")

                # Add to results
                for detection in detections:
                    detection['filename'] = video_file.name
                    detection['filepath'] = str(video_file)
                    self.results.extend([detection])

            except Exception as e:
                print(f"❌ Error processing {video_file.name}: {e}")

    def save_results(self, output_path):
        """Save results to CSV file"""
        if not self.results:
            print("❌ No results to save")
            return

        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to: {output_path}")

        # Print summary statistics
        print(f"\n📈 Results Summary:")
        print(f"   Total detections: {len(df)}")
        print(f"   Unique files: {df['filename'].nunique()}")
        print(f"   Average detection confidence: {df['detection_confidence'].mean():.3f}")
        print(f"   Average OCR confidence: {df['ocr_confidence'].mean():.3f}")

        # Show most common detected texts
        if len(df) > 0:
            print(f"\n🔤 Most Common Detected Plates:")
            top_plates = df['text'].value_counts().head(10)
            for plate, count in top_plates.items():
                print(f"   {plate}: {count} times")

    def create_summary_report(self, output_dir):
        """Create a summary report with visualizations"""
        if not self.results:
            print("❌ No results to create report")
            return

        df = pd.DataFrame(self.results)
        report_path = Path(output_dir) / "batch_processing_report.txt"

        with open(report_path, 'w') as f:
            f.write("Indian License Plate Recognition - Batch Processing Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Detections: {len(df)}\n")
            f.write(f"Unique Files: {df['filename'].nunique()}\n")
            f.write(f"Average Detection Confidence: {df['detection_confidence'].mean():.3f}\n")
            f.write(f"Average OCR Confidence: {df['ocr_confidence'].mean():.3f}\n\n")

            f.write("Top 20 Detected License Plates:\n")
            f.write("-" * 40 + "\n")
            top_plates = df['text'].value_counts().head(20)
            for plate, count in top_plates.items():
                f.write(f"{plate:<15} : {count} times\n")

            f.write("\n\nConfidence Distribution:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Detection Confidence > 0.8: {(df['detection_confidence'] > 0.8).sum()} detections\n")
            f.write(f"Detection Confidence > 0.5: {(df['detection_confidence'] > 0.5).sum()} detections\n")
            f.write(f"OCR Confidence > 0.8: {(df['ocr_confidence'] > 0.8).sum()} detections\n")
            f.write(f"OCR Confidence > 0.5: {(df['ocr_confidence'] > 0.5).sum()} detections\n")

        print(f"📋 Summary report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Batch Process Indian License Plate Recognition')
    parser.add_argument('--input', '-i', required=True, help='Input directory path')
    parser.add_argument('--output', '-o', default='batch_results', help='Output directory path')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLOv8 model path')
    parser.add_argument('--confidence', '-c', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--type', '-t', choices=['images', 'videos', 'both'], default='images', 
                       help='Process images, videos, or both')
    parser.add_argument('--extensions', nargs='+', help='File extensions to process')
    parser.add_argument('--report', action='store_true', help='Generate summary report')

    args = parser.parse_args()

    print("🚗 Indian License Plate Recognition - Batch Processor")
    print("=" * 60)

    # Initialize batch processor
    processor = BatchProcessor(args.model, args.confidence)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process based on type
    if args.type in ['images', 'both']:
        print("\n📸 Processing Images...")
        extensions = args.extensions or ['jpg', 'jpeg', 'png', 'bmp']
        processor.process_images_in_directory(args.input, args.output, extensions)

    if args.type in ['videos', 'both']:
        print("\n🎥 Processing Videos...")
        extensions = args.extensions or ['mp4', 'avi', 'mov', 'mkv']
        processor.process_videos_in_directory(args.input, args.output, extensions)

    # Save results
    results_csv = os.path.join(args.output, 'batch_results.csv')
    processor.save_results(results_csv)

    # Generate report if requested
    if args.report:
        processor.create_summary_report(args.output)

    print(f"\n✅ Batch processing completed!")
    print(f"📁 Check results in: {args.output}")

if __name__ == "__main__":
    main()