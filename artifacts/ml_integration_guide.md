# Machine Learning Workflow Integration Guide

This guide outlines how to integrate ML workflows with your GDOT traffic analysis system, leveraging your RTX A5000 GPUs.

## 1. Environment Setup

```bash
# Install ML-specific dependencies
pip install tensorboard pyyaml scikit-learn torchmetrics pandas
```

## 2. Project Structure Enhancement

Add these directories to your existing project:
```
gdot_traffic_analysis/
├── ml/
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Model evaluation
│   ├── visualize.py      # Training visualization
│   ├── augment.py        # Data augmentation utilities
│   └── export.py         # Model export utilities
├── datasets/
│   ├── raw/              # Original video frames
│   ├── labeled/          # Annotated datasets
│   ├── splits/           # Train/val/test splits
│   └── augmented/        # Augmented training data
└── experiments/
    ├── runs/             # Training runs
    ├── models/           # Saved models
    ├── configs/          # Training configurations
    └── results/          # Evaluation results
```

## 3. Core ML Workflows

### 3.1 Data Collection Pipeline

Create `data_extraction.py`:

```python
#!/usr/bin/env python3
"""
Extract training frames from GDOT traffic videos
"""
import argparse
import cv2
import os
import numpy as np
from pathlib import Path
import time
import random
import shutil

def extract_frames(video_path, output_dir, sample_rate=30, max_frames=1000):
    """Extract frames from video for training data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    
    print(f"Video: {video_path}")
    print(f"Duration: {duration:.2f}s ({total_frames} frames at {fps} fps)")
    
    # Determine frame extraction strategy
    if total_frames <= max_frames:
        # Extract every sample_rate frame
        frame_indices = range(0, total_frames, sample_rate)
    else:
        # Random sampling
        frame_indices = sorted(random.sample(range(total_frames), max_frames))
    
    # Extract frames
    count = 0
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Save frame
        output_path = output_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        count += 1
        
        # Progress update
        if count % 100 == 0:
            print(f"Extracted {count} frames")
    
    cap.release()
    print(f"Completed: {count} frames extracted to {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract frames from traffic videos')
    parser.add_argument('--video', required=True, help='Input video file or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--sample-rate', type=int, default=30, help='Frame sampling rate')
    parser.add_argument('--max-frames', type=int, default=1000, help='Maximum frames per video')
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if video_path.is_file():
        # Process a single video file
        extract_frames(str(video_path), args.output, args.sample_rate, args.max_frames)
    elif video_path.is_dir():
        # Process all videos in directory
        for ext in ['*.mp4', '*.avi', '*.mov']:
            for video_file in video_path.glob(ext):
                # Create subdirectory for each video
                video_name = video_file.stem
                output_subdir = Path(args.output) / video_name
                extract_frames(str(video_file), output_subdir, args.sample_rate, args.max_frames)
    else:
        print(f"Error: {args.video} is not a valid file or directory")
        return False

if __name__ == "__main__":
    main()
```

### 3.2 Data Annotation Guide

1. Install CVAT or use LabelImg for annotation:

```bash
# For LabelImg
pip install labelImg

# Launch LabelImg
labelImg datasets/raw datasets/labeled/labels YOLO
```

2. Create a data configuration YAML:

```yaml
# datasets/traffic.yaml
train: datasets/splits/train
val: datasets/splits/val
test: datasets/splits/test

# Classes
nc: 4  # Number of classes
names: ['car', 'motorcycle', 'bus', 'truck']

# Class mapping to COCO dataset
# This helps leverage pre-trained weights
map: [2, 3, 5, 7]
```

### 3.3 Training Pipeline

Create `ml/train.py`:

```python
#!/usr/bin/env python3
"""
YOLOv8 training pipeline for traffic analysis
"""
import argparse
import os
import yaml
import time
import torch
from pathlib import Path
from ultralytics import YOLO
import shutil
import json

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--data', required=True, help='Data configuration YAML')
    parser.add_argument('--model', default='yolov8n.pt', help='Base model path')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--device', default='', help='cuda device (e.g. 0 or 0,1)')
    parser.add_argument('--project', default='experiments/runs', help='Project directory')
    parser.add_argument('--name', default='exp', help='Experiment name')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--freeze', type=int, default=0, help='Freeze first n layers')
    args = parser.parse_args()

    # Create experiment directory
    time_str = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_{time_str}"
    run_dir = Path(args.project) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Copy data configuration
    shutil.copy(args.data, run_dir / 'data.yaml')
    
    # Setup GPU training
    if args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    # Load YOLO model
    model = YOLO(args.model)
    
    # Print GPU info
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Training on {device_count} GPU(s):")
        for i in range(device_count):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"     Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"     CUDA Capability: {props.major}.{props.minor}")
    else:
        print("No GPU available, training on CPU")
        
    # Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.img_size,
        workers=args.workers,
        project=args.project,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        cos_lr=True,
        close_mosaic=10,
        augment=True
    )
    
    # Evaluate on test set
    results = model.val()
    
    # Export to ONNX for deployment
    model.export(format='onnx', dynamic=True)
    
    print(f"Training complete. Results saved to {run_dir}")
    
if __name__ == "__main__":
    main()
```

### 3.4 Model Evaluation

Create `ml/evaluate.py`:

```python
#!/usr/bin/env python3
"""
Evaluate trained model on test data
"""
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2

def evaluate_model(model_path, data_yaml, output_dir, conf=0.25, iou=0.5):
    """Evaluate model and generate performance metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        conf=conf,
        iou=iou,
        verbose=True
    )
    
    # Save summary metrics
    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1-score": float(results.box.map50 * 2 / (results.box.mp + results.box.mr + 1e-6)),
        "inference_time": float(results.speed["inference"]),
        "confidence_threshold": conf,
        "iou_threshold": iou
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualizations
    
    # 1. Confusion matrix
    if hasattr(results, "confusion_matrix"):
        cm = results.confusion_matrix.matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()
    
    # 2. Precision-Recall curve
    if hasattr(results.box, "p") and hasattr(results.box, "r"):
        plt.figure(figsize=(9, 6))
        plt.plot(results.box.r, results.box.p)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(output_dir / "precision_recall.png")
        plt.close()
    
    # 3. F1 curve
    if hasattr(results.box, "f1"):
        plt.figure(figsize=(9, 6))
        plt.plot(results.box.c, results.box.f1)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Confidence Threshold')
        plt.grid(True)
        plt.savefig(output_dir / "f1_curve.png")
        plt.close()
    
    print(f"Evaluation metrics saved to {output_dir}")
    return metrics

def visualize_predictions(model_path, test_dir, output_dir, conf=0.25, max_samples=10):
    """Generate visualization of model predictions on test images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Get test images
    test_dir = Path(test_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_dir.glob(f"*{ext}")))
    
    # Limit number of samples
    if len(test_images) > max_samples:
        test_images = np.random.choice(test_images, max_samples, replace=False)
    
    # Generate predictions
    for img_path in test_images:
        results = model(str(img_path), conf=conf)[0]
        
        # Get annotated image
        annotated_img = results.plot()
        
        # Save output
        output_path = output_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), annotated_img)
    
    print(f"Prediction visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 model')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Data configuration YAML')
    parser.add_argument('--test-dir', help='Directory with test images for visualization')
    parser.add_argument('--output', default='experiments/results', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--visualize-samples', type=int, default=10, 
                      help='Number of test samples to visualize')
    args = parser.parse_args()
    
    # Create output directory with timestamp
    time_str = Path(args.model).stem
    output_dir = Path(args.output) / time_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate model metrics
    metrics = evaluate_model(
        args.model,
        args.data,
        output_dir / "metrics",
        args.conf,
        args.iou
    )
    
    # Visualize predictions on test samples
    if args.test_dir:
        visualize_predictions(
            args.model,
            args.test_dir,
            output_dir / "visualizations",
            args.conf,
            args.visualize_samples
        )
    
    print("Evaluation complete")
    print(f"mAP50: {metrics['mAP50']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")

if __name__ == "__main__":
    main()
```

### 3.5 Model Optimization

Create `ml/optimize.py`:

```python
#!/usr/bin/env python3
"""
Optimize YOLOv8 models for inference
"""
import argparse
import torch
import time
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
from ultralytics import YOLO

def benchmark_model(model, img_size=640, batch_size=1, num_iterations=100, warmup=10):
    """Benchmark model inference speed"""
    device = next(model.parameters()).device
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(warmup):
        _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = model(dummy_input)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    latency_ms = (elapsed_time / num_iterations) * 1000
    fps = num_iterations / elapsed_time
    
    return {
        'latency_ms': latency_ms,
        'fps': fps,
        'batch_size': batch_size,
        'img_size': img_size
    }

def export_onnx(model, img_size=640, batch_size=1, output_path='model.onnx'):
    """Export model to ONNX format"""
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        opset_version=12,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    return output_path

def optimize_onnx(onnx_path, output_path=None):
    """Optimize ONNX model for inference"""
    if output_path is None:
        output_path = Path(onnx_path).with_suffix('.opt.onnx')
    
    # Create ONNX Runtime session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create InferenceSession
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    # Get model metadata
    metadata = session.get_modelmeta()
    
    # Save optimized model
    # Note: in a real implementation, you would use ONNX Runtime's
    # model optimization tools directly. This is a simplified example.
    model = onnx.load(onnx_path)
    onnx.save(model, output_path)
    
    return output_path

def benchmark_onnx(onnx_path, img_size=640, batch_size=1, num_iterations=100, warmup=10):
    """Benchmark ONNX model inference speed"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Create dummy input
    dummy_input = np.random.randn(batch_size, 3, img_size, img_size).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = session.run(None, {input_name: dummy_input})
        
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    latency_ms = (elapsed_time / num_iterations) * 1000
    fps = num_iterations / elapsed_time
    
    return {
        'latency_ms': latency_ms,
        'fps': fps,
        'batch_size': batch_size,
        'img_size': img_size
    }

def quantize_model(model_path, quantized_path=None, method='dynamic'):
    """Quantize PyTorch model to int8 or fp16"""
    if quantized_path is None:
        quantized_path = Path(model_path).with_suffix('.quantized.pt')
        
    # Load model
    model = YOLO(model_path)
    
    if method == 'dynamic':
        # Dynamic quantization (easier but less optimized)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
    elif method == 'static':
        # Static quantization (requires calibration data)
        # This is a simplified example and would need to be expanded
        # with proper calibration in a real implementation
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        # Here you would run calibration data through the model
        torch.quantization.convert(model, inplace=True)
        quantized_model = model
    elif method == 'fp16':
        # Half precision (FP16)
        quantized_model = model.half()
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), quantized_path)
    
    return quantized_path

def main():
    parser = argparse.ArgumentParser(description='Optimize YOLOv8 model for inference')
    parser.add_argument('--model', required=True, help='Path to YOLOv8 model')
    parser.add_argument('--output-dir', default='experiments/optimized', help='Output directory')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for benchmarking')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    parser.add_argument('--quantize-method', choices=['dynamic', 'static', 'fp16'], 
                      default='fp16', help='Quantization method')
    parser.add_argument('--export-onnx', action='store_true', help='Export to ONNX')
    parser.add_argument('--optimize-onnx', action='store_true', help='Optimize ONNX model')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark models')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    model_path = args.model
    base_model = YOLO(model_path)
    
    # Initial benchmark
    if args.benchmark:
        print(f"Benchmarking base model: {model_path}")
        benchmark_results = benchmark_model(
            base_model.model,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_iterations=args.iterations
        )
        print(f"Base model: {benchmark_results['latency_ms']:.2f}ms per inference "
              f"({benchmark_results['fps']:.2f} FPS)")
    
    # Quantize model
    if args.quantize:
        quantized_path = output_dir / f"{Path(model_path).stem}_{args.quantize_method}.pt"
        print(f"Quantizing model using {args.quantize_method} method...")
        quantize_model(model_path, quantized_path, args.quantize_method)
        print(f"Quantized model saved to {quantized_path}")
        
        # Benchmark quantized model
        if args.benchmark:
            quantized_model = YOLO(quantized_path)
            benchmark_results = benchmark_model(
                quantized_model.model,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_iterations=args.iterations
            )
            print(f"Quantized model: {benchmark_results['latency_ms']:.2f}ms per inference "
                 f"({benchmark_results['fps']:.2f} FPS)")
    
    # Export to ONNX
    if args.export_onnx:
        onnx_path = output_dir / f"{Path(model_path).stem}.onnx"
        print(f"Exporting model to ONNX: {onnx_path}")
        base_model.export(format='onnx', imgsz=args.img_size)
        print(f"ONNX model exported to {onnx_path}")
        
        # Optimize ONNX model
        if args.optimize_onnx:
            optimized_path = output_dir / f"{Path(model_path).stem}.opt.onnx"
            print(f"Optimizing ONNX model: {optimized_path}")
            optimize_onnx(onnx_path, optimized_path)
            print(f"Optimized ONNX model saved to {optimized_path}")
            
            # Benchmark ONNX models
            if args.benchmark:
                print("Benchmarking ONNX models...")
                onnx_results = benchmark_onnx(
                    onnx_path,
                    img_size=args.img_size,
                    batch_size=args.batch_size,
                    num_iterations=args.iterations
                )
                print(f"ONNX model: {onnx_results['latency_ms']:.2f}ms per inference "
                     f"({onnx_results['fps']:.2f} FPS)")
                
                opt_results = benchmark_onnx(
                    optimized_path,
                    img_size=args.img_size,
                    batch_size=args.batch_size,
                    num_iterations=args.iterations
                )
                print(f"Optimized ONNX model: {opt_results['latency_ms']:.2f}ms per inference "
                     f"({opt_results['fps']:.2f} FPS)")
    
    print("Model optimization complete")

if __name__ == "__main__":
    main()
```

## 4. Integration with Existing Code

Modify `analyzer.py` to support model switching and performance tracking:

```python
# Add these imports
from pathlib import Path
import json
import time

# Add to TrafficAnalyzer __init__
self.model_stats = {
    "model_path": model_path,
    "inference_times": [],
    "avg_inference_time": 0,
    "max_inference_time": 0,
    "min_inference_time": float('inf'),
    "frames_processed": 0,
    "detections_count": 0
}

# In the process method, add performance tracking:
# Before running detection
infer_start = time.time()

# After detection completes
infer_time = (time.time() - infer_start) * 1000  # ms
self.model_stats["inference_times"].append(infer_time)
self.model_stats["frames_processed"] += 1
self.model_stats["detections_count"] += len(detections)
self.model_stats["avg_inference_time"] = np.mean(self.model_stats["inference_times"])
self.model_stats["max_inference_time"] = max(self.model_stats["max_inference_time"], infer_time)
self.model_stats["min_inference_time"] = min(self.model_stats["min_inference_time"], infer_time)

# Add a method to save model statistics
def save_model_stats(self, output_path):
    """Save model performance statistics to file"""
    stats = self.model_stats.copy()
    
    # Convert numpy values to native Python types for JSON
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            stats[key] = value.tolist()
        elif isinstance(value, np.generic):
            stats[key] = value.item()
    
    # Add summary statistics
    if stats["inference_times"]:
        stats["percentile_90"] = float(np.percentile(stats["inference_times"], 90))
        stats["percentile_95"] = float(np.percentile(stats["inference_times"], 95))
        stats["percentile_99"] = float(np.percentile(stats["inference_times"], 99))
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Model statistics saved to {output_path}")
```

## 5. Running the ML Workflow

1. **Data Collection**:
```bash
mkdir -p datasets/raw
python data_extraction.py --video recordings/gdot_stream_20250411_061303.mp4 --output datasets/raw
```

2. **Data Annotation**:
Using LabelImg or CVAT, annotate the extracted frames and organize into:
```
datasets/labeled/
├── images/        # Image files
└── labels/        # YOLO format annotations
```

3. **Data Splitting**:
```bash
mkdir -p datasets/splits/{train,val,test}
# 70% train, 20% val, 10% test (implement your own split script or use sklearn)
```

4. **Model Training**:
```bash
python ml/train.py --data datasets/traffic.yaml --epochs 50 --batch 8
```

5. **Model Evaluation**:
```bash
python ml/evaluate.py --model experiments/runs/exp_20250424_120000/weights/best.pt --data datasets/traffic.yaml
```

6. **Model Optimization**:
```bash
python ml/optimize.py --model experiments/runs/exp_20250424_120000/weights/best.pt --quantize --quantize-method fp16 --export-onnx --benchmark
```

7. **Deploying Optimized Model**:
```bash
python main.py --source "https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/ATL-CCTV-0092/playlist.m3u8" --model experiments/optimized/best_fp16.pt --dashboard
```

## 6. Troubleshooting

### CUDA Memory Issues
- Reduce batch size: `--batch 4` or lower
- Use smaller model: `yolov8n.pt` instead of larger variants
- Clear cache: Add `torch.cuda.empty_cache()` before large operations
- Monitor memory: `watch -n 1 nvidia-smi`

### Training Problems
- Learning rate issues: Try `--lr 0.001` for finer control
- Overfitting: Add `--augment` for more data augmentation
- Slow training: Enable mixed precision with `--half`

### Export Issues
- ONNX compatibility: Install `pip install onnx onnxruntime-gpu`
- TensorRT: For maximum performance, install NVIDIA TensorRT

### Inference Optimization
- Thread pinning: `taskset -c 0-7 python main.py` (assigns specific CPU cores)
- NUMA awareness: `numactl --cpunodebind=0 --membind=0 python main.py`
- GPU optimization: `CUDA_VISIBLE_DEVICES=0 python main.py`
