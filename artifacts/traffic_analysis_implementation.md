# Traffic Analysis Implementation Plan

## 1. Environment Setup

```bash
# Create project structure
mkdir -p gdot_ml/{models,data,output,logs}
cd gdot_ml

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy matplotlib
pip install supervision
```

## 2. Project Structure

```
gdot_ml/
├── main.py           # Entry point
├── analyzer.py       # Core traffic analysis functionality  
├── visualizer.py     # Dashboard visualization
├── utils.py          # Helper functions
├── models/           # YOLOv8 weights
│   └── yolov8n.pt    # Pre-trained model
├── data/             # Sample videos
└── output/           # Results
```

## 3. Implementation Files

### main.py
```python
#!/usr/bin/env python3
"""
Main entry point for Traffic Analysis system
"""
import argparse
import torch
import os
from pathlib import Path
from analyzer import TrafficAnalyzer
from visualizer import DashboardVisualizer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Traffic Analysis System')
    parser.add_argument('--source', required=True, help='Path to video file or GDOT stream URL')
    parser.add_argument('--model', default='models/yolov8n.pt', help='Path to YOLOv8 model')
    parser.add_argument('--output', default='output/result.mp4', help='Output video path')
    parser.add_argument('--dashboard', action='store_true', help='Show real-time dashboard')
    parser.add_argument('--conf', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default=None, 
                      help='Device to run on (cuda device, i.e. 0 or cpu)')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Check for CUDA
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Print GPU info if using CUDA
    if device.startswith('cuda'):
        cuda_id = 0 if device == 'cuda' else int(device.split(':')[1])
        torch.cuda.set_device(cuda_id)
        print(f"Using GPU: {torch.cuda.get_device_name(cuda_id)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available memory: {torch.cuda.get_device_properties(cuda_id).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")
    
    # Initialize visualizer if dashboard is enabled
    visualizer = None
    if args.dashboard:
        visualizer = DashboardVisualizer()
    
    # Initialize traffic analyzer
    analyzer = TrafficAnalyzer(
        model_path=args.model,
        device=device,
        conf=args.conf
    )
    
    # Process video
    analyzer.process(args.source, args.output, visualizer)
    
    # Generate statistics report
    report_path = Path(args.output).with_suffix('.json')
    analyzer.save_stats(str(report_path))
    print(f"Statistics saved to {report_path}")

if __name__ == "__main__":
    main()
```

### analyzer.py
```python
"""
Traffic analysis core functionality
"""
import cv2
import torch
import json
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
from collections import defaultdict
import time

class TrafficAnalyzer:
    def __init__(self, model_path='models/yolov8n.pt', device=None, conf=0.3):
        """
        Initialize traffic analyzer with YOLOv8 and ByteTrack
        
        Args:
            model_path: Path to YOLOv8 model
            device: Device to use (cuda or cpu)
            conf: Confidence threshold for detections
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf = conf
        
        # Traffic classes (from COCO dataset)
        self.traffic_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        
        # Tracking
        self.tracker = sv.ByteTrack()
        
        # Analysis data
        self.vehicle_counts = {cls_name: 0 for cls_name in self.class_names.values()}
        self.speed_data = {}  # track_id -> speed estimate
        self.flow_rate = []  # Vehicles per minute
        self.inference_times = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metrics
        self.total_frames = 0
        self.processed_frames = 0
        self.start_time = None
        self.fps_avg = 0
        
    def process(self, source, output_path, visualizer=None):
        """
        Process video source and track vehicles
        
        Args:
            source: Path to video file or stream URL
            output_path: Path to save output video
            visualizer: Optional dashboard visualizer
        """
        # Open video source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open {source}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize counter (horizontal line at 2/3 height)
        self.counter = sv.LineZone(
            start=sv.Point(0, int(height * 2/3)),
            end=sv.Point(width, int(height * 2/3))
        )
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )
        
        # Create output video writer
        output = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Start timer
        self.start_time = time.time()
        frame_count = 0
        last_time = self.start_time
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.total_frames += 1
                
                # Process every N frames (N=3 for better performance)
                if frame_count % 3 == 0:
                    # Measure inference time
                    infer_start = time.time()
                    
                    # Run detection with GPU acceleration and mixed precision
                    with torch.cuda.amp.autocast(enabled=self.device.startswith('cuda')):
                        results = self.model(
                            frame,
                            conf=self.conf,
                            classes=self.traffic_classes,
                            device=self.device
                        )[0]
                    
                    infer_time = (time.time() - infer_start) * 1000  # ms
                    self.inference_times.append(infer_time)
                    
                    # Convert results to supervision format
                    detections = sv.Detections.from_ultralytics(results)
                    
                    # Track objects
                    detections = self.tracker.update(detections=detections)
                    
                    # Count vehicles crossing the line
                    self.counter.trigger(detections=detections)
                    
                    # Update metrics
                    self._update_metrics(detections, frame_count, fps)
                    
                    # Annotate frame with detections and metrics
                    annotated_frame = self._annotate_frame(frame, detections, infer_time)
                    
                    # Update dashboard if available
                    if visualizer:
                        visualizer.update(
                            frame=annotated_frame,
                            counts=self.vehicle_counts,
                            speeds=self.speed_data,
                            flow=self.flow_rate,
                            inference_time=np.mean(self.inference_times[-100:])
                        )
                    
                    # Write to output video
                    output.write(annotated_frame)
                    
                    # Calculate FPS
                    if frame_count % 30 == 0:
                        now = time.time()
                        self.fps_avg = 30 / (now - last_time) if (now - last_time) > 0 else 0
                        last_time = now
                        
                        # Print progress
                        if total_frames > 0:
                            progress = frame_count / total_frames * 100
                            print(f"Progress: {progress:.1f}% | FPS: {self.fps_avg:.1f} | Inference: {np.mean(self.inference_times[-100:]):.1f}ms")
                    
                    self.processed_frames += 1
                    
                frame_count += 1
                    
        except KeyboardInterrupt:
            print("Processing interrupted by user.")
        finally:
            # Release resources
            cap.release()
            output.release()
            cv2.destroyAllWindows()
            
            # Print summary
            duration = time.time() - self.start_time
            print(f"Processing complete: {output_path}")
            print(f"Processed {self.processed_frames} frames in {duration:.1f}s")
            print(f"Average inference time: {np.mean(self.inference_times):.1f}ms")
            print(f"Vehicle counts: {self.vehicle_counts}")
            
            return True
    
    def _update_metrics(self, detections, frame_count, fps):
        """Update vehicle counts, speed estimates, and flow rate"""
        # Update vehicle counts
        for class_id in self.class_names:
            self.vehicle_counts[self.class_names[class_id]] = self.counter.count(class_id=class_id)
        
        # Track positions for speed estimation (simplified)
        # In a real implementation, this would use temporal data and camera calibration
        for i, (xyxy, _, _, track_id, _) in enumerate(detections):
            if track_id == None:
                continue
                
            # Calculate center point
            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Store position for track
            if track_id not in self.speed_data:
                self.speed_data[track_id] = {
                    "positions": [(center_x, center_y)],
                    "frames": [frame_count],
                    "speed": 0
                }
            else:
                self.speed_data[track_id]["positions"].append((center_x, center_y))
                self.speed_data[track_id]["frames"].append(frame_count)
                
                # Calculate speed if we have multiple positions
                if len(self.speed_data[track_id]["positions"]) >= 2:
                    # Calculate displacement
                    prev_x, prev_y = self.speed_data[track_id]["positions"][-2]
                    curr_x, curr_y = self.speed_data[track_id]["positions"][-1]
                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    displacement = np.sqrt(dx*dx + dy*dy)
                    
                    # Calculate time difference in frames
                    frame_diff = self.speed_data[track_id]["frames"][-1] - self.speed_data[track_id]["frames"][-2]
                    
                    # Calculate speed in pixels per frame and convert to "mph" for visualization
                    # In a real implementation, calibration would be needed
                    speed = displacement / frame_diff
                    mph = speed * 5  # Arbitrary scaling for visualization
                    
                    self.speed_data[track_id]["speed"] = mph
                    
        # Calculate flow rate (vehicles per minute)
        minute_frames = fps * 60
        if frame_count % minute_frames == 0:
            total = sum(self.vehicle_counts.values())
            self.flow_rate.append(total)
        
    def _annotate_frame(self, frame, detections, infer_time):
        """Add bounding boxes, tracking IDs, and metrics to frame"""
        # Format custom labels
        labels = []
        for class_id, confidence, track_id in zip(detections.class_id, detections.confidence, detections.tracker_id):
            if class_id in self.class_names and track_id is not None:
                speed = 0
                if track_id in self.speed_data:
                    speed = self.speed_data[track_id]["speed"]
                    
                label = f"{self.class_names[class_id]}: {int(speed)} mph"
                labels.append(label)
            else:
                labels.append("")
                
        # Add bounding boxes and tracking IDs
        annotated_frame = self.box_annotator.annotate(frame.copy(), detections, labels)
        
        # Add counting line
        annotated_frame = self.line_annotator.annotate(annotated_frame, self.counter)
        
        # Add info text
        cv2.putText(
            annotated_frame, 
            f"Inference: {infer_time:.1f}ms | FPS: {self.fps_avg:.1f}",
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Add counts
        y = 70
        for label, count in self.vehicle_counts.items():
            cv2.putText(
                annotated_frame,
                f"{label}: {count}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            y += 30
            
        return annotated_frame
        
    def save_stats(self, output_path):
        """Save analysis statistics to JSON file"""
        stats = {
            "timestamp": self.timestamp,
            "vehicle_counts": self.vehicle_counts,
            "flow_rate": self.flow_rate,
            "inference_times": {
                "mean": float(np.mean(self.inference_times)),
                "min": float(np.min(self.inference_times)),
                "max": float(np.max(self.inference_times))
            },
            "fps": float(self.fps_avg),
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "processing_time": time.time() - self.start_time if self.start_time else 0
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
```

### visualizer.py
```python
"""
Real-time visualization dashboard
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class DashboardVisualizer:
    def __init__(self, window_name="Traffic Analysis Dashboard"):
        """
        Initialize the visualization dashboard
        
        Args:
            window_name: Name of the OpenCV window
        """
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        
        # Create figure for plots
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Store data for plots
        self.counts_history = []
        self.speed_history = []
        self.flow_history = []
        self.inference_history = []
        
    def update(self, frame, counts, speeds, flow, inference_time):
        """
        Update the dashboard with new data
        
        Args:
            frame: Current video frame with annotations
            counts: Vehicle count dictionary
            speeds: Vehicle speed dictionary
            flow: Traffic flow rate data
            inference_time: Current inference time
        """
        # Update data history
        self.counts_history.append(counts.copy())
        self.inference_history.append(inference_time)
        
        # Extract speed values
        speed_values = [data["speed"] for data in speeds.values() if "speed" in data]
        if speed_values:
            self.speed_history.append(speed_values)
        
        if flow:
            self.flow_history = flow.copy()
            
        # Limit history length
        max_history = 100
        if len(self.counts_history) > max_history:
            self.counts_history = self.counts_history[-max_history:]
        if len(self.inference_history) > max_history:
            self.inference_history = self.inference_history[-max_history:]
        if len(self.speed_history) > max_history:
            self.speed_history = self.speed_history[-max_history:]
            
        # Create dashboard
        dashboard = self._create_dashboard(frame)
        
        # Display
        cv2.imshow(self.window_name, dashboard)
        cv2.waitKey(1)
        
    def _create_dashboard(self, frame):
        """Create the combined dashboard with frame and plots"""
        # Create blank canvas
        dashboard = np.zeros((800, 1200, 3), dtype=np.uint8)
        dashboard.fill(240)  # Light gray background
        
        # Resize video frame to fit dashboard
        h, w = frame.shape[:2]
        scale = min(500/h, 600/w)
        resized_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        
        # Place frame on dashboard
        y_offset = 50
        x_offset = 30
        dashboard[y_offset:y_offset+resized_frame.shape[0], 
                 x_offset:x_offset+resized_frame.shape[1]] = resized_frame
        
        # Add title
        cv2.putText(
            dashboard,
            "GDOT Traffic Analysis Dashboard",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        
        # Clear previous plots
        for ax in self.axs.flat:
            ax.clear()
            
        # Plot 1: Vehicle Counts (Bar Chart)
        if self.counts_history:
            latest_counts = self.counts_history[-1]
            classes = list(latest_counts.keys())
            values = list(latest_counts.values())
            
            colors = ['#2196f3', '#ff9800', '#4caf50', '#9c27b0']
            self.axs[0, 0].bar(classes, values, color=colors)
            self.axs[0, 0].set_title('Vehicle Counts by Type')
            self.axs[0, 0].set_ylabel('Count')
            
        # Plot 2: Speed Distribution (Histogram)
        if self.speed_history:
            all_speeds = [speed for speeds in self.speed_history[-5:] for speed in speeds]
            if all_speeds:
                self.axs[0, 1].hist(all_speeds, bins=10, color='#2196f3', alpha=0.7)
                self.axs[0, 1].set_title('Speed Distribution (mph)')
                self.axs[0, 1].set_xlabel('Speed')
                self.axs[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Traffic Flow Rate
        if len(self.flow_history) > 1:
            x = range(len(self.flow_history))
            self.axs[1, 0].plot(x, self.flow_history, '-o', color='#2196f3')
            self.axs[1, 0].set_title('Traffic Flow Rate')
            self.axs[1, 0].set_xlabel('Time (min)')
            self.axs[1, 0].set_ylabel('Vehicles per minute')
            
        # Plot 4: Inference Performance
        if self.inference_history:
            x = range(len(self.inference_history))
            self.axs[1, 1].plot(x, self.inference_history, color='#ff9800')
            self.axs[1, 1].set_title('Inference Time')
            self.axs[1, 1].set_xlabel('Frame')
            self.axs[1, 1].set_ylabel('Time (ms)')
            
        # Adjust layout
        self.fig.tight_layout(pad=3.0)
        
        # Convert matplotlib figure to OpenCV image
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        plot_img = np.array(canvas.buffer_rgba())
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        
        # Resize plots to fit dashboard
        plot_img = cv2.resize(plot_img, (580, 650))
        
        # Place plots on dashboard
        dashboard[y_offset:y_offset+plot_img.shape[0], 
                 x_offset+resized_frame.shape[1]+20:x_offset+resized_frame.shape[1]+20+plot_img.shape[1]] = plot_img
        
        return dashboard
    
    def close(self):
        """Close the dashboard window"""
        cv2.destroyWindow(self.window_name)
        plt.close(self.fig)
```

### utils.py
```python
"""
Utility functions for traffic analysis
"""
import os
import cv2
import numpy as np
import torch

def download_model(model_name='yolov8n.pt', save_dir='models'):
    """
    Download YOLOv8 model if not present
    
    Args:
        model_name: Name of the model (yolov8n.pt, yolov8s.pt, etc)
        save_dir: Directory to save the model
    
    Returns:
        Path to the model
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name)
    
    if not os.path.exists(model_path):
        from ultralytics import YOLO
        model = YOLO(model_name)
        model.export(format="onnx")  # This will download the model first
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Model already exists at {model_path}")
        
    return model_path

def check_gpu_compatibility():
    """
    Check GPU compatibility and return device settings
    
    Returns:
        device: torch device to use
        cuda_info: dictionary of CUDA information
    """
    cuda_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    if cuda_info["available"]:
        # Get properties for all GPUs
        devices = []
        for i in range(cuda_info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / 1e9,
                "sm_count": props.multi_processor_count
            })
        cuda_info["devices"] = devices
        
        # Use first CUDA device
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        
    return device, cuda_info

def create_video_writer(source, output_path, width=None, height=None):
    """
    Create a VideoWriter object based on source properties
    
    Args:
        source: Source video path or capture object
        output_path: Output video path
        width: Custom width (optional)
        height: Custom height (optional)
    
    Returns:
        VideoWriter object
    """
    # Get video properties from source
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
    else:
        cap = source
        
    # Get original dimensions
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use custom dimensions if provided
    out_width = width if width else orig_width
    out_height = height if height else orig_height
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output_path, 
        fourcc, 
        fps, 
        (out_width, out_height)
    )
    
    # If source was opened here, close it
    if isinstance(source, str):
        cap.release()
        
    return writer
```

## 4. Running the Code

Download a pre-trained YOLOv8 model:
```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
```

Run with recorded video:
```bash
python main.py --source data/traffic_sample.mp4 --dashboard
```

Run with GDOT stream:
```bash
python main.py --source "https://sfs-msc-pub-lq-01.navigator.dot.ga.gov:443/rtplive/ATL-CCTV-0092/playlist.m3u8" --dashboard
```

## 5. Next Steps for Enhancement

1. **Improve speed estimation**: Implement proper camera calibration
2. **Add lane detection**: Identify lane boundaries and track lane-specific metrics
3. **Add anomaly detection**: Identify traffic incidents and unusual patterns
4. **Fine-tune model**: Collect and label Georgia-specific traffic data for transfer learning
5. **Develop persistent database**: Store metrics over time for historical analysis
6. **Add web interface**: Create browser-based dashboard for remote monitoring
