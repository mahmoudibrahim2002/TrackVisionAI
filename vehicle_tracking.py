 import numpy as np
from collections import deque, defaultdict
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO
import supervision as sv
import cv2
import argparse
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
import shutil
from datetime import datetime
import glob
import subprocess
import json
import tempfile
import os
os.environ['CUDA_CACHE_PATH'] = os.path.expanduser('~/.nv')
os.environ['NUMBAPRO_CUDA_DRIVER'] = '/usr/lib/x86_64-linux-gnu/libcuda.so'
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice'

# Initialize argument parser
parser = argparse.ArgumentParser(description='Vehicle Tracking System')
parser.add_argument('--input_video', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_dir', type=str, default='reports', help='Directory to save reports')
parser.add_argument('--config', type=str, help='Path to JSON config file')
args = parser.parse_args()

# Load configuration if provided
if args.config:
    with open(args.config) as f:
        config = json.load(f)
    
    SOURCE = np.array(config['SOURCE'])
    TARGET_WIDTH = config['TARGET_WIDTH']
    TARGET_HEIGHT = config['TARGET_HEIGHT']
    lower_line_source = np.array(config['lower_line'])
    upper_line_source = np.array(config['upper_line'])
else:
    # Default values if no config provided
    SOURCE = np.array([[700, 787], [2298, 803], [5039, 2159], [-550, 2159]])
    TARGET_WIDTH = 20
    TARGET_HEIGHT = 180
    lower_line_source = np.array([[700, 787], [2298, 803]])
    upper_line_source = np.array([[150, 1606], [3750, 1606]])

# Define target ROI
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

# Update paths based on command-line arguments
SOURCE_VIDEO_PATH = args.input_video
REPORTS_DIR = args.output_dir
TARGET_VIDEO_PATH = os.path.join(REPORTS_DIR, "output.mp4")

# Ensure output directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# Initialize license plate detection model
license_plate_model = YOLO("license_plate_detectorV2.pt")
    
# Vehicle Class Mapping
VEHICLE_CLASSES = {
    1: 'bicycle', 2: 'car', 3: 'motorcycle',
    5: 'bus', 7: 'truck',  # Add more as needed from your YOLO model
}

# Dictionary to store the best frame for each vehicle's license plate
vehicle_reports = defaultdict(dict)  # {tracker_id: {"best_frame": frame, "best_score": score, "speed": speed}}

# View Transformer class
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Initialize view transformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# Transform the lines to the target coordinate system
lower_line_target = view_transformer.transform_points(lower_line_source)
upper_line_target = view_transformer.transform_points(upper_line_source)

# Extract the Y-coordinates of the transformed lines
lower_line_y = lower_line_target[0][1]  # Y-coordinate of the lower line
upper_line_y = upper_line_target[0][1]  # Y-coordinate of the upper line

# Function to evaluate license plate visibility
def evaluate_license_plate(frame, bbox):
    # Convert bbox coordinates to integers
    x1, y1, x2, y2 = [int(coord) for coord in bbox]

    # Ensure coordinates are within frame bounds
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    # Check if ROI is valid
    if x2 <= x1 or y2 <= y1:
        return 0, None, None

    vehicle_roi = frame[y1:y2, x1:x2]

    # Detect license plates in the vehicle ROI
    lp_results = license_plate_model(vehicle_roi)[0]
    lp_detections = sv.Detections.from_ultralytics(lp_results)

    if len(lp_detections) == 0:
        return 0, None, None  # No license plate detected

    # Get the best license plate detection (highest confidence)
    best_idx = np.argmax(lp_detections.confidence)
    best_lp = lp_detections.xyxy[best_idx]
    lp_confidence = lp_detections.confidence[best_idx]

    # Extract license plate coordinates (relative to vehicle ROI)
    lp_x1, lp_y1, lp_x2, lp_y2 = map(int, best_lp)

    # Add padding around license plate
    padding = 5
    lp_x1 = max(0, lp_x1 - padding)
    lp_y1 = max(0, lp_y1 - padding)
    lp_x2 = min(vehicle_roi.shape[1], lp_x2 + padding)
    lp_y2 = min(vehicle_roi.shape[0], lp_y2 + padding)

    # Crop license plate from vehicle ROI
    lp_image = vehicle_roi[lp_y1:lp_y2, lp_x1:lp_x2]

    return lp_confidence, vehicle_roi, lp_image

def sanitize_text(text):
    """Remove or replace problematic characters"""
    if not isinstance(text, str):
        return str(text)
    return text.encode('ascii', errors='replace').decode('ascii').replace('?', '-')

# Function to generate PDF report for a vehicle
def generate_vehicle_report(tracker_id, vehicle_data, output_dir="reports"):
    """
    Generate a comprehensive PDF report for a tracked vehicle with:
    - Vehicle details (type, model, speed)
    - License plate recognition
    - Visual evidence (vehicle and plate images)
    - Violation information
    """
    
    # Ensure all required fields exist with proper defaults
    required_fields = {
        'vehicle_type': 'Unknown',
        'direction': 'Unknown',
        'speed': 0,
        'entry_time': datetime.now(),
        'exit_time': datetime.now(),
        'violation': False,
        'license_plate_text': None,
        'best_image': None,
        'license_plate_image': None
    }
    
    for field, default in required_fields.items():
        if field not in vehicle_data:
            vehicle_data[field] = default

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for images
    temp_dir = os.path.join(output_dir, "temp_images")
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize PDF with custom settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Function to safely encode text for Arial
    def safe_text(text):
        if text is None:
            return ""
        try:
            # First try encoding as latin-1
            return str(text).encode('latin-1', errors='strict').decode('latin-1')
        except UnicodeEncodeError:
            # Fallback to ASCII with replacement for unsupported chars
            return str(text).encode('ascii', errors='replace').decode('ascii').replace('?', '-')
    
    # --- Header Section ---
    pdf.set_font('Arial', 'B', 16)
    
    # Color coding based on violation status
    if vehicle_data['violation']:
        pdf.set_fill_color(255, 220, 220)  # Light red
        pdf.set_text_color(200, 0, 0)      # Dark red
        title = safe_text(f"TRAFFIC VIOLATION REPORT - #{tracker_id}")
    else:
        pdf.set_fill_color(220, 230, 255)  # Light blue
        pdf.set_text_color(0, 0, 120)      # Dark blue
        title = safe_text(f"VEHICLE REPORT - #{tracker_id}")
    
    # Header with colored background
    pdf.cell(0, 10, title, 0, 1, 'C', 1)
    pdf.set_text_color(0, 0, 0)  # Reset to black text
    pdf.ln(8)
    
    # --- Vehicle Details Section ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, safe_text("VEHICLE DETAILS"), 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Create details table
    with pdf.table() as table:
        # Vehicle Identification
        row = table.row()
        row.cell(safe_text("Vehicle ID:"), border=1)
        row.cell(safe_text(f"#{tracker_id}"), border=1)
        
        # Vehicle Type
        row = table.row()
        row.cell(safe_text("Vehicle Type:"), border=1)
        row.cell(safe_text(vehicle_data['vehicle_type'].title()), border=1)
                
        # Direction
        row = table.row()
        row.cell(safe_text("Direction:"), border=1)
        row.cell(safe_text(vehicle_data['direction']), border=1)
        
        # Speed (highlight violations)
        row = table.row()
        row.cell(safe_text("Speed:"), border=1)
        speed_text = safe_text(f"{int(vehicle_data['speed'])} km/h")
        if vehicle_data['violation']:
            pdf.set_text_color(255, 0, 0)
            speed_text = safe_text(f"{int(vehicle_data['speed'])} km/h (VIOLATION)")
        row.cell(speed_text, border=1)
        pdf.set_text_color(0, 0, 0)  # Reset color
        
        # Timestamps
        row = table.row()
        row.cell(safe_text("Entry Time:"), border=1)
        row.cell(safe_text(vehicle_data['entry_time'].strftime('%Y-%m-%d %H:%M:%S')), border=1)
        
        row = table.row()
        row.cell(safe_text("Exit Time:"), border=1)
        row.cell(safe_text(vehicle_data['exit_time'].strftime('%Y-%m-%d %H:%M:%S')), border=1)
    
    pdf.ln(12)
    
    # --- Visual Evidence Section ---
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 80, 180)  # Blue text
    pdf.cell(200, 10, safe_text("VISUAL EVIDENCE"), 0, 1, 'C')
    
    # Add decorative line
    pdf.set_draw_color(0, 80, 180)
    pdf.set_line_width(0.8)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(8)
    
    # Vehicle Image
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 8, safe_text("VEHICLE IMAGE"), 0, 1, 'C')
    
    # Handle case where no vehicle image was captured
    if vehicle_data['best_image'] is None:
        vehicle_data['best_image'] = np.zeros((100, 100, 3), dtype=np.uint8)
        vehicle_data['best_image'][:] = (200, 200, 200)  # Gray background
        cv2.putText(vehicle_data['best_image'], "No Image Available", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save and add vehicle image to PDF
    vehicle_img_path = os.path.join(temp_dir, f"vehicle_{tracker_id}.jpg")
    cv2.imwrite(vehicle_img_path, cv2.cvtColor(vehicle_data['best_image'], cv2.COLOR_RGB2BGR))
    
    # Image with shadow effect
    pdf.set_fill_color(200, 200, 200)
    pdf.rect(53, pdf.get_y()+3, 104, 79, 'F')
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(1)
    pdf.rect(50, pdf.get_y(), 100, 75)
    pdf.image(vehicle_img_path, x=52, y=pdf.get_y()+2, w=96, h=71)
    pdf.ln(80)
    
    # --- License Plate Section ---
    if vehicle_data.get('license_plate_image') is not None:
        # Save license plate image to temp file
        lp_img_path = os.path.join(temp_dir, f"lp_{tracker_id}.jpg")
        cv2.imwrite(lp_img_path, cv2.cvtColor(vehicle_data['license_plate_image'], cv2.COLOR_RGB2BGR))
        
        # License Plate Header
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 8, safe_text("LICENSE PLATE"), 0, 1, 'C')
        
        # Calculate proportional dimensions
        h, w = vehicle_data['license_plate_image'].shape[:2]
        aspect_ratio = h / w
        display_width = 100
        display_height = int(display_width * aspect_ratio)
        
        # License plate image with frame
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(1)
        pdf.rect(50, pdf.get_y(), 100, display_height + 4)
        pdf.image(lp_img_path, x=52, y=pdf.get_y()+2, w=96, h=display_height)
        pdf.ln(display_height + 10)
        
        # Recognized Text Display
        if vehicle_data.get('license_plate_text'):
            plate_text = safe_text(vehicle_data['license_plate_text'].upper())
            pdf.set_fill_color(220, 255, 220)  # Light green background
            pdf.rect(40, pdf.get_y(), 120, 12, 'F')
            pdf.set_font('Arial', 'B', 14)
            pdf.set_text_color(0, 100, 0)  # Dark green text
            pdf.cell(200, 10, safe_text(f"RECOGNIZED: {plate_text}"), 0, 1, 'C')
        else:
            pdf.set_fill_color(255, 220, 220)  # Light red background
            pdf.rect(40, pdf.get_y(), 120, 10, 'F')
            pdf.set_font('Arial', 'I', 10)
            pdf.set_text_color(150, 0, 0)  # Dark red text
            pdf.cell(200, 8, safe_text("NO LICENSE PLATE RECOGNIZED"), 0, 1, 'C')
        pdf.ln(8)
        
    # --- Footer ---
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, safe_text(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), 0, 1, 'C')
    
    # Save PDF
    pdf_path = os.path.join(output_dir, f"report_{tracker_id}.pdf")
    pdf.output(pdf_path)
    
    # Cleanup temporary files
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")
    
    return pdf_path

# Smoothing function using a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Smoothing function using exponential smoothing
def exponential_smoothing(data, alpha):
    smoothed_data = [data[0]]
    for i in range(1, len(data)):
        smoothed_data.append(alpha * data[i] + (1 - alpha) * smoothed_data[i - 1])
    return smoothed_data

# Kalman Filter for speed estimation
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_value = 0
        self.estimation_error = 1

    def update(self, measurement):
        # Prediction
        self.estimation_error += self.process_variance

        # Update
        kalman_gain = self.estimation_error / (self.estimation_error + self.measurement_variance)
        self.estimated_value += kalman_gain * (measurement - self.estimated_value)
        self.estimation_error *= (1 - kalman_gain)

        return self.estimated_value

# Initialize YOLO model
model = YOLO("yolo11x.pt")

# Video information and frame generator
video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)

# ByteTrack tracker initialization
CONFIDENCE_THRESHOLD = 0.4
byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=CONFIDENCE_THRESHOLD)

# Annotators configuration
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
bounding_box_annotator = sv.BoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps, position=sv.Position.BOTTOM_CENTER, color_lookup=sv.ColorLookup.TRACK)

# Polygon zone for detection filtering
polygon_zone = sv.PolygonZone(polygon=SOURCE)
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# Transform the lines to the target coordinate system
lower_line_target = view_transformer.transform_points(lower_line_source)
upper_line_target = view_transformer.transform_points(upper_line_source)

# Extract the Y-coordinates of the transformed lines
lower_line_y = lower_line_target[0][1]  # Y-coordinate of the lower line
upper_line_y = upper_line_target[0][1]  # Y-coordinate of the upper line

# Dictionary to store coordinates for each tracked vehicle
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps * 2))  # Larger window for smoothing

# Dictionary to store Kalman filters for each tracker_id
kalman_filters = {}

# Define counting variables
up_count = 0
down_count = 0

# Define traffic level thresholds
TRAFFIC_THRESHOLDS = {"Low": 5, "Moderate": 15}  # Vehicles per minute

def classify_traffic(count):
    if count <= TRAFFIC_THRESHOLDS["Low"]:
        return "Low", (0, 255, 0)  # Green
    elif count <= TRAFFIC_THRESHOLDS["Moderate"]:
        return "Moderate", (0, 255, 255)  # Yellow
    else:
        return "Heavy", (0, 0, 255)  # Red

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global up_count, down_count

    # Calculate progress percentage
    progress = int((index / video_info.total_frames) * 100)
    if progress % 5 == 0 or index == 0 or index == video_info.total_frames - 1:
        print(f"Progress: {progress}%")
        print(f"Status: Processing frame {index}/{video_info.total_frames}")

    # Perform object detection using YOLO
    result = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Filter detections by confidence and class (excluding persons)
    detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
    detections = detections[detections.class_id != 0]

    # Filter detections outside the polygon zone
    detections = detections[polygon_zone.trigger(detections)]

    # Apply non-max suppression to refine detections
    detections = detections.with_nms(0.5)

    # Update detections with ByteTrack tracker
    detections = byte_track.update_with_detections(detections=detections)

    # Get bottom center coordinates of detections
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

    # Transform points to the target ROI
    points = view_transformer.transform_points(points=points).astype(int)

    # Store coordinates and check for best license plate visibility
    for idx, (tracker_id, [_, y]) in enumerate(zip(detections.tracker_id, points)):
        if tracker_id is None:
            continue  # Skip detections without tracker IDs

        bbox = detections.xyxy[idx]
        class_id = detections.class_id[idx]
        vehicle_type = VEHICLE_CLASSES.get(int(class_id), 'unknown')

        # Initialize vehicle report if not exists
        if tracker_id not in vehicle_reports:
            print(f"New vehicle detected: ID #{tracker_id}")
            vehicle_reports[tracker_id] = {
                'best_score': 0,
                'best_frame': None,
                'best_image': None,          # This will be the final image used for both API and report
                'final_image_saved': False,  # Flag to ensure we process image only once
                'license_plate_image': None,
                'license_plate_text': None,
                'speed': 0,
                'max_speed': 0,
                'direction': None,
                'vehicle_type': vehicle_type,
                'entry_time': datetime.now(),
                'exit_time': None,
                'violation': False,
                'frames_count': 0,
                'api_called': False
            }

        # Increment frame count
        vehicle_reports[tracker_id]['frames_count'] += 1

        # Evaluate license plate visibility
        lp_score, vehicle_image, lp_image = evaluate_license_plate(frame, bbox)

        # Store coordinates for speed calculation
        if tracker_id not in coordinates:
            coordinates[tracker_id] = deque(maxlen=video_info.fps * 2)

        prev_y = coordinates[tracker_id][-1] if coordinates[tracker_id] else None
        coordinates[tracker_id].append(y)

        # Update counting logic and direction
        if prev_y is not None:
            if prev_y > lower_line_y >= y:
                up_count += 1
                vehicle_reports[tracker_id]['direction'] = "Northbound"
            elif prev_y < upper_line_y <= y:
                down_count += 1
                vehicle_reports[tracker_id]['direction'] = "Southbound"

        # Update best frame if current license plate is more visible
        if lp_score > vehicle_reports[tracker_id]['best_score']:
            vehicle_reports[tracker_id]['best_score'] = lp_score
            
            # Save the definitive vehicle image that will be used for both API and report
            x1, y1, x2, y2 = map(int, bbox)
            vehicle_reports[tracker_id]['best_image'] = cv2.cvtColor(
                frame[y1:y2, x1:x2],
                cv2.COLOR_BGR2RGB
            )
            
            # Store license plate image if detected
            if lp_score > 0.7 and lp_image is not None:
                vehicle_reports[tracker_id]['license_plate_image'] = cv2.cvtColor(
                    lp_image, cv2.COLOR_BGR2RGB
                )

    # Process vehicles that are leaving the frame
    active_tracker_ids = set(tid for tid in detections.tracker_id if tid is not None)
    for tracker_id in list(vehicle_reports.keys()):
        if tracker_id not in active_tracker_ids:
            # Only process if we have sufficient data
            if (vehicle_reports[tracker_id]['frames_count'] > 10 and
                vehicle_reports[tracker_id]['direction'] is not None and
                vehicle_reports[tracker_id]['speed'] is not None):
                
                # Call API if we haven't already and we have a good image
                if (not vehicle_reports[tracker_id]['api_called'] and 
                    vehicle_reports[tracker_id]['best_image'] is not None):
                    
                    try:
                        # Save the definitive vehicle image
                        final_img_path = os.path.join(REPORTS_DIR, f"final_vehicle_{tracker_id}.jpg")
                        cv2.imwrite(final_img_path, cv2.cvtColor(
                            vehicle_reports[tracker_id]['best_image'],
                            cv2.COLOR_RGB2BGR
                        ))
                        
                        print(f"Calling API for vehicle #{tracker_id} with final image")
                        result = subprocess.run(
                            ["python", "number_plate_redaction.py", 
                             "--api-key", "5d8990dd9403136b4801aa4f06f600695644bfc2",
                             final_img_path],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if result.returncode == 0:
                            output = json.loads(result.stdout)
                            
                            if output and len(output) > 0 and "results" in output[0]:
                                best_plate = None
                                best_score = 0
                                vehicle_type = "Unknown"
                                vehicle_model = "Unknown"
                                
                                for plate_data in output[0]["results"]:
                                    if plate_data["score"] > best_score:
                                        best_plate = plate_data["plate"].upper()
                                        best_score = plate_data["score"]
                                        vehicle_type = plate_data["vehicle"].get("type", "Unknown")
                                
                                if best_plate:
                                    vehicle_reports[tracker_id].update({
                                        'license_plate_text': best_plate,
                                        'vehicle_type': vehicle_type,
                                        'api_called': True
                                    })
                                    print(f"Final API results for #{tracker_id}: {best_plate} | {vehicle_type}")
                    
                    except Exception as e:
                        print(f"API call failed for #{tracker_id}: {str(e)}")
                    finally:
                        # Clean up temp file
                        try:
                            os.remove(final_img_path)
                        except:
                            pass
                
                # Generate report using the same image that was sent to API
                vehicle_reports[tracker_id]['exit_time'] = datetime.now()
                print(f"Generating report for vehicle #{tracker_id}")
                generate_vehicle_report(tracker_id, vehicle_reports[tracker_id])
                
                # Clean up tracking data
                if tracker_id in coordinates:
                    del coordinates[tracker_id]
                if tracker_id in kalman_filters:
                    del kalman_filters[tracker_id]
                del vehicle_reports[tracker_id]
                
    # Calculate and update speeds in vehicle reports
    for tracker_id in detections.tracker_id:
        if tracker_id is None:
            continue

        if len(coordinates.get(tracker_id, [])) >= video_info.fps / 2:
            y_coords = list(coordinates[tracker_id])
            distances = np.abs(np.diff(y_coords))
            times = np.arange(len(y_coords)) / video_info.fps
            speeds = distances / np.diff(times) * 3.6  # Convert to km/h

            # Apply smoothing
            smoothed_speeds = moving_average(speeds, window_size=5)

            # Initialize Kalman filter if needed
            if tracker_id not in kalman_filters:
                kalman_filters[tracker_id] = KalmanFilter(process_variance=1e-5, measurement_variance=0.1)

            # Update speed in vehicle report
            if len(smoothed_speeds) > 0:
                filtered_speed = kalman_filters[tracker_id].update(np.mean(smoothed_speeds))
                vehicle_reports[tracker_id]['speed'] = filtered_speed
                vehicle_reports[tracker_id]['max_speed'] = max(
                    vehicle_reports[tracker_id]['max_speed'],
                    filtered_speed
                )

                # Mark as violation if speed exceeds threshold
                if filtered_speed > 95:
                    vehicle_reports[tracker_id]['violation'] = True

    # Generate reports for vehicles that have exited the frame
    active_tracker_ids = set(tid for tid in detections.tracker_id if tid is not None)
    for tracker_id in list(vehicle_reports.keys()):
        if tracker_id not in active_tracker_ids:
            # Only generate report if we have minimum data
            if (vehicle_reports[tracker_id]['frames_count'] > 5 and  # Seen in at least 5 frames
                vehicle_reports[tracker_id]['direction'] is not None and
                vehicle_reports[tracker_id]['speed'] is not None):

                vehicle_reports[tracker_id]['exit_time'] = datetime.now()
                print(f"Generating report for vehicle #{tracker_id}")
                generate_vehicle_report(tracker_id, vehicle_reports[tracker_id])
            else:
                print(f"Insufficient data for vehicle #{tracker_id} - frames: {vehicle_reports[tracker_id]['frames_count']}")

            # Clean up tracking data
            if tracker_id in coordinates:
                del coordinates[tracker_id]
            if tracker_id in kalman_filters:
                del kalman_filters[tracker_id]
            del vehicle_reports[tracker_id]

    # Prepare labels and identify violations
    labels = []
    violation_indices = []
    for idx, tracker_id in enumerate(detections.tracker_id):
        if tracker_id is None:
            labels.append("")
            continue
            
        if len(coordinates.get(tracker_id, [])) < video_info.fps / 2:
            labels.append(f"#{tracker_id}")
        else:
            # Calculate speed using multiple points - SAME METHOD AS BASIC VERSION
            y_coords = list(coordinates[tracker_id])
            distances = np.abs(np.diff(y_coords))
            times = np.arange(len(y_coords)) / video_info.fps
            speeds = distances / np.diff(times) * 3.6  # Convert to km/h

            # Apply smoothing (moving average)
            smoothed_speeds = moving_average(speeds, window_size=5)

            # Initialize Kalman filter if needed
            if tracker_id not in kalman_filters:
                kalman_filters[tracker_id] = KalmanFilter(process_variance=1e-5, measurement_variance=0.1)

            # Get filtered speed
            filtered_speed = kalman_filters[tracker_id].update(np.mean(smoothed_speeds))
            
            # Update vehicle report with speed
            vehicle_reports[tracker_id]['speed'] = filtered_speed
            vehicle_reports[tracker_id]['max_speed'] = max(
                vehicle_reports[tracker_id]['max_speed'],
                filtered_speed
            )
            
            # Mark as violation if speed exceeds threshold
            if filtered_speed > 95:
                vehicle_reports[tracker_id]['violation'] = True
                violation_indices.append(idx)

            labels.append(f"#{tracker_id} {int(filtered_speed)} km/h")

    # Annotate the frame
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    # Draw red boxes for violations
    for idx in violation_indices:
        x1, y1, x2, y2 = detections.xyxy[idx].astype(int)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Add traces and labels
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # Draw boundary lines
    annotated_frame = sv.draw_line(
        scene=annotated_frame,
        start=sv.Point(x=int(lower_line_source[0][0]), y=int(lower_line_source[0][1])),
        end=sv.Point(x=int(lower_line_source[1][0]), y=int(lower_line_source[1][1])),
        color=sv.Color(0, 255, 0),
        thickness=2
    )

    annotated_frame = sv.draw_line(
        scene=annotated_frame,
        start=sv.Point(x=int(upper_line_source[0][0]), y=int(upper_line_source[0][1])),
        end=sv.Point(x=int(upper_line_source[1][0]), y=int(upper_line_source[1][1])),
        color=sv.Color(0, 0, 255),
        thickness=2
    )

    # Display traffic information
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (30, 20), (520, 220), (50, 50, 50), -1)
    alpha = 0.6
    annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)
    cv2.rectangle(annotated_frame, (30, 20), (520, 220), (255, 255, 255), 2)

    current_vehicle_count = len(detections)
    traffic_level, color = classify_traffic(current_vehicle_count)
    violations = len(violation_indices)

    cv2.putText(annotated_frame, f"Northbound: {up_count}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Southbound: {down_count}", (50, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Vehicles: {current_vehicle_count}", (50, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Traffic: {traffic_level}", (50, 170), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Violations: {violations}", (50, 210), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return annotated_frame

def main():
    parser = argparse.ArgumentParser(description='Vehicle Tracking System')
    parser.add_argument('--input_video', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='reports')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # Load coordinates from config file
    with open(args.config) as f:
        config = json.load(f)
    
    SOURCE = np.array(config['SOURCE'])
    TARGET_WIDTH = config['TARGET_WIDTH']
    TARGET_HEIGHT = config['TARGET_HEIGHT']
    lower_line_source = np.array(config['lower_line'])
    upper_line_source = np.array(config['upper_line'])

    # Define target ROI
    TARGET = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ])

    print("Starting processing with configuration:")
    print(f"Source points: {SOURCE.tolist()}")
    print(f"Lower line: {lower_line_source.tolist()}")
    print(f"Upper line: {upper_line_source.tolist()}")

    # Initialize view transformer
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)


    print(f"Starting vehicle tracking on video: {SOURCE_VIDEO_PATH}")
    print(f"Reports will be saved to: {REPORTS_DIR}")
    print(f"Processed video will be saved to: {TARGET_VIDEO_PATH}")

    # Initialize progress tracking
    print("Progress: 0%")
    print("Status: Initializing video processing")

    try:
        # Process video
        sv.process_video(
            source_path=SOURCE_VIDEO_PATH,
            target_path=TARGET_VIDEO_PATH,
            callback=callback
        )

        # Verify output
        if os.path.exists(TARGET_VIDEO_PATH):
           print(f"Successfully created video: {TARGET_VIDEO_PATH}")
           print(f"File size: {os.path.getsize(TARGET_VIDEO_PATH)} bytes")
        else:
           print(f"Failed to create output video at {TARGET_VIDEO_PATH}")

        # Verify video was created
        if not os.path.exists(TARGET_VIDEO_PATH):
            raise FileNotFoundError(f"Output video not created at {TARGET_VIDEO_PATH}")

        # Verify video is playable
        cap = cv2.VideoCapture(TARGET_VIDEO_PATH)
        if not cap.isOpened():
            raise ValueError("Output video file is corrupted or unreadable")
        cap.release()

        report_count = len(glob.glob(os.path.join(REPORTS_DIR, "report_*.pdf")))
        print(f"Processing complete! Generated {report_count} reports")
        print(f"Output video saved to: {TARGET_VIDEO_PATH}")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise


if __name__ == "__main__":
    main()