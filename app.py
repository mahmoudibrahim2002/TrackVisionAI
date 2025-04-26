import streamlit as st
import os
import glob
import pandas as pd
import PyPDF2
import base64
import re
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
import subprocess
import threading
import queue
import sys
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx
import tempfile
from pathlib import Path
import plotly.express as px
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Configuration
REPORTS_DIR = "reports"
OUTPUT_VIDEO_PATH = os.path.join(REPORTS_DIR, "output.mp4")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Progress Queue for real-time updates
progress_queue = queue.Queue()

# Page Config (must be first)
st.set_page_config(
    page_title="Vehicle Tracking System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - Dark mode compatible
st.markdown("""
<style>
    /* Common styles */
    :root {
        --primary: #3a86ff;
        --secondary: #2ecc71;
        --danger: #ff5a5f;
        --dark: #2b2d42;
        --light: #f8f9fa;
        --gray: #adb5bd;
        --bg-dark: #121212;
        --card-dark: #1e1e1e;
        --text-dark: #e0e0e0;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stApp {
        background-color: var(--bg-dark);
    }
    
    /* Dark mode text colors */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, 
    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, 
    .stMarkdown h6, .stMarkdown a, .stMarkdown span, .stMarkdown div {
        color: var(--text-dark) !important;
    }
    
    /* Introduction page styles */
    .intro-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, var(--primary), var(--dark));
        color: white;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .intro-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .intro-subtitle {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    
    .feature-card {
        background: var(--card-dark);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border-left: 4px solid var(--primary);
        transition: transform 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-card h3 {
        color: var(--primary);
        margin-top: 0;
    }
    
    .tech-badge {
        display: inline-block;
        background: rgba(58, 134, 255, 0.2);
        color: var(--primary);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .profile-card {
        background: var(--card-dark);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .profile-img {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto 1rem;
        border: 5px solid var(--primary);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .profile-title {
        font-size: 1.5rem;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .profile-subtitle {
        color: var(--gray);
        margin-bottom: 1.5rem;
    }
    
    .social-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--primary);
        color: white !important;
        margin: 0 0.5rem;
        transition: all 0.3s;
        text-decoration: none;
    }
    
    .social-link:hover {
        background: var(--dark);
        transform: translateY(-3px);
    }
    
    .nav-button {
        display: block;
        width: 100%;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: var(--primary);
        color: white !important;
        text-align: center;
        font-weight: 600;
        transition: all 0.3s;
        text-decoration: none;
        border: none;
        cursor: pointer;
    }
    
    .nav-button:hover {
        background: var(--dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Side image layout */
    .side-image-container {
        display: flex;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .side-image {
        width: 35%;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        object-fit: cover;
        align-self: flex-start;
    }
    
    .side-content {
        width: 65%;
    }
    
    /* Analysis page styles */
    .metric-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        background: var(--card-dark);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--gray);
    }
    
    .violation-true {
        color: var(--danger);
        font-weight: bold;
    }
    
    .violation-false {
        color: var(--secondary);
        font-weight: bold;
    }
    
    .verified-badge {
        background-color: var(--secondary);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Progress bar styles */
    .progress-container {
        width: 100%;
        background-color: #333;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 20px;
        background-color: var(--primary);
        border-radius: 10px;
        width: 0%;
        transition: width 0.3s;
        text-align: center;
        color: white;
        font-size: 12px;
        line-height: 20px;
    }
    
    .upload-btn {
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    
    .upload-btn:hover {
        background-color: var(--dark) !important;
        transform: translateY(-2px) !important;
    }
            
                .celebration {
        animation: celebrate 2s ease infinite;
        text-align: center;
        font-size: 1.5rem;
        color: #FF6B6B;
        margin: 1rem 0;
    }
    
    @keyframes celebrate {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    
    .video-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .video-box {
        width: 48%;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 10px;
        background: #2b2d42;
    }
    
    .video-title {
        text-align: center;
        margin-bottom: 10px;
        color: white;
    }
        /* Enhanced Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d) !important;
        padding: 1rem !important;
        border-radius: 0 20px 20px 0 !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Sidebar Navigation Buttons */
    .stButton button {
        width: 100% !important;
        padding: 12px !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
        margin: 8px 0 !important;
    }
    
    .stButton button:hover {
        background: rgba(255,255,255,0.2) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        border-color: rgba(255,255,255,0.5) !important;
    }
    
    /* Active Button State */
    .stButton button:focus:not(:active) {
        background: rgba(255,255,255,0.3) !important;
        border-color: white !important;
        color: white !important;
    }
    
    /* Analysis Page Improvements */
    .metric-card {
        background: rgba(30, 30, 30, 0.7) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        border-left: 4px solid #3a86ff !important;
        transition: all 0.3s !important;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3) !important;
    }
    
    /* Enhanced Data Table */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Video Container Styling */
    .video-container {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    }
    
    /* PDF Viewer Styling */
    .pdf-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-top: 2rem;
    }
        /* Gradient matching the title */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #3a86ff, #2b2d42) !important;
        padding: 1rem !important;
        border-radius: 0 20px 20px 0 !important;
        box-shadow: 5px 0 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Sidebar text contrast */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Title gradient */
    .intro-title {
        background: linear-gradient(135deg, #3a86ff, #2b2d42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
    }
</style>
""", unsafe_allow_html=True)

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def try_load_image(path, default_url=None):
    try:
        return Image.open(path)
    except:
        if default_url:
            return load_image_from_url(default_url)
        return None

def extract_value(text, patterns, default='N/A', is_number=False):
    """Extract value using multiple possible patterns"""
    if not isinstance(patterns, list):
        patterns = [patterns]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = match.group(1).strip()
                if is_number:
                    return int(value)
                return value
            except (IndexError, ValueError):
                continue
    return default

def extract_all_fields(text):
    """Extract all required fields from PDF text"""
    patterns = {
        'tracker_id': [
            r"VEHICLE\s*ID:\s*#?(\w+)",
            r"ID:\s*(\w+)",
            r"Tracker\s*ID:\s*(\w+)"
        ],
        'vehicle_type': [
            r"VEHICLE\s*TYPE:\s*([^\n]+)",
            r"Type:\s*([^\n]+)",
            r"Vehicle:\s*([^\n]+)"
        ],
        'speed': [
            r"SPEED:\s*(\d+)\s*km/h",
            r"Speed:\s*(\d+)",
            r"(\d+)\s*km/h",
            r"Speed\s*\(\s*km/h\s*\):\s*(\d+)"
        ],
        'violation': [
            r"(VIOLATION)",
            r"(Violation)",
            r"(Speeding)"
        ],
        'entry_time': [
            r"ENTRY\s*TIME:\s*([^\n]+)",
            r"Entry:\s*([^\n]+)",
            r"Time\s*in:\s*([^\n]+)"
        ],
        'exit_time': [
            r"EXIT\s*TIME:\s*([^\n]+)",
            r"Exit:\s*([^\n]+)",
            r"Time\s*out:\s*([^\n]+)"
        ],
        'license_plate': [
            r"RECOGNIZED:\s*([A-Z0-9]+)\s",  # Match only the plate before any space
            r"LICENSE\s*PLATE:\s*([A-Z0-9]+)\s",
            r"Plate:\s*([A-Z0-9]+)\s",
            r"([A-Z]{2}\s?\d{4}\s?[A-Z]?)",
            r"Number:\s*([A-Z0-9]+)\s"
        ]
    }
    
    # Extract license plate with special handling
    license_plate = 'N/A'
    for pattern in patterns['license_plate']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            license_plate = match.group(1).strip().replace(" ", "").upper()
            # Remove any non-alphanumeric characters that might have been captured
            license_plate = re.sub(r'[^A-Z0-9]', '', license_plate)
            # Ensure we don't capture "REPORTGENERATED" or other metadata
            if len(license_plate) <= 10:  # Reasonable max length for license plates
                break
            else:
                license_plate = 'N/A'
    
    data = {
        'tracker_id': extract_value(text, patterns['tracker_id']),
        'vehicle_type': extract_value(text, patterns['vehicle_type']),
        'speed': extract_value(text, patterns['speed'], 0, True),
        'violation': bool(extract_value(text, patterns['violation'])),
        'entry_time': extract_value(text, patterns['entry_time']),
        'exit_time': extract_value(text, patterns['exit_time']),
        'license_plate': license_plate,
        'verified': True  # Automatically mark all as verified
    }
    
    return data

def load_data():
    pdf_files = glob.glob(os.path.join(REPORTS_DIR, "*.pdf"))
    data = []
    
    for pdf_path in pdf_files:
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
                
                entry = extract_all_fields(text)
                entry['file_name'] = os.path.basename(pdf_path)
                
                # Extract vehicle type and model with better patterns
                type_match = re.search(r"Vehicle Type:\s*([^\n]+)", text)
                model_match = re.search(r"Vehicle Model:\s*([^\n]+)", text)
                
                if type_match:
                    entry['vehicle_type'] = type_match.group(1).strip()
                else:
                    entry['vehicle_type'] = "Unknown"
                    
                if model_match:
                    entry['vehicle_model'] = model_match.group(1).strip()
                else:
                    entry['vehicle_model'] = "Unknown"
                
                data.append(entry)
        except Exception as e:
            st.error(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
            continue
    
    return pd.DataFrame(data)


def display_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            st.markdown(f"""
            <div class="pdf-container">
                <iframe src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%" 
                        height="600" 
                        style="border:none;">
                </iframe>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not display PDF: {str(e)}")

def run_vehicle_tracking(video_path):
    try:
        # Create temp directory
        temp_dir = os.path.join(REPORTS_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded video
        temp_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_path.read())
        
        # Clear previous reports and output
        for f in glob.glob(os.path.join(REPORTS_DIR, "report_*.pdf")):
            os.remove(f)
        if os.path.exists(OUTPUT_VIDEO_PATH):
            os.remove(OUTPUT_VIDEO_PATH)
        
        # Run processing
        command = [
            sys.executable,
            "vehicle_tracking.py",
            "--input_video", temp_video_path,
            "--output_dir", REPORTS_DIR
        ]
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor process output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                if "Progress:" in output:
                    try:
                        progress = int(output.split("Progress:")[1].strip().replace("%", ""))
                        progress_queue.put(("progress", progress))
                    except:
                        pass
                elif "Generated report for vehicle" in output:
                    progress_queue.put(("report", output.strip()))
                else:
                    progress_queue.put(("status", output.strip()))
        
        # Check for errors
        stderr = process.stderr.read()
        if process.returncode != 0:
            progress_queue.put(("error", stderr))
            return False
        
        report_count = len(glob.glob(os.path.join(REPORTS_DIR, "report_*.pdf")))
        progress_queue.put(("complete", report_count))
        return True
        
    except Exception as e:
        progress_queue.put(("error", str(e)))
        return False

def process_video():
    """Handle the complete video processing workflow with coordinates"""
    # Validate we have required data
    if 'video_bytes' not in st.session_state or 'coordinates_config' not in st.session_state:
        st.error("Missing video data or coordinates configuration!")
        return
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Save video to temp file
        video_path = os.path.join(temp_dir, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(st.session_state['video_bytes'])
        
        # Save coordinates config to temp file
        config_path = os.path.join(temp_dir, "tracking_config.json")
        with open(config_path, "w") as f:
            json.dump(st.session_state.coordinates_config, f)
        
        # Create UI placeholders
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        video_placeholder = st.empty()
        result_placeholder = st.empty()
        
        # Build processing command
        command = [
            sys.executable,
            "vehicle_tracking.py",
            "--input_video", video_path,
            "--output_dir", REPORTS_DIR,
            "--config", config_path
        ]
        
        # Start processing in separate thread
        thread = threading.Thread(
            target=run_processing,
            args=(command, progress_placeholder, status_placeholder, video_placeholder)
        )
        
        # Add streamlit context if needed
        if hasattr(threading.current_thread(), '_st_script_run_ctx'):
            add_script_run_ctx(thread)
        
        thread.start()
        
        # Monitor progress
        while thread.is_alive():
            update_progress_ui(progress_placeholder, status_placeholder, video_placeholder)
            time.sleep(0.1)
        
        # Final update
        update_progress_ui(progress_placeholder, status_placeholder, video_placeholder)
        thread.join()
        
        # Display results
        if os.path.exists(OUTPUT_VIDEO_PATH):
            with video_placeholder.container():
                st.markdown("### Processed Output")
                with open(OUTPUT_VIDEO_PATH, "rb") as f:
                    st.video(f.read())
            
            report_count = len(glob.glob(os.path.join(REPORTS_DIR, "report_*.pdf")))
            result_placeholder.success(f"Processing complete! Generated {report_count} reports.")
        
    except Exception as e:
        error_msg = str(e)
        # Handle encoding errors specifically
        if "ordinal not in range" in error_msg or "UnicodeEncodeError" in error_msg:
            st.error("Character encoding error in report generation. Please check your vehicle data for special characters.")
        else:
            st.error(f"Processing failed: {error_msg}")
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Clear processing flags
        if 'confirmed' in st.session_state:
            del st.session_state['confirmed']

def run_processing(command, progress_placeholder, status_placeholder, video_placeholder):
    """Run the processing command"""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output = output.strip()
                if "Progress:" in output:
                    try:
                        progress = int(output.split("Progress:")[1].strip().replace("%", ""))
                        progress_queue.put(("progress", progress))
                    except:
                        pass
                elif "Generated report for vehicle" in output:
                    progress_queue.put(("report", output))
                elif output:  # Only add non-empty output
                    progress_queue.put(("status", output))
        
        # Check for errors
        stderr = process.stderr.read()
        if process.returncode != 0:
            progress_queue.put(("error", stderr))
            return False
        
        report_count = len(glob.glob(os.path.join(REPORTS_DIR, "report_*.pdf")))
        progress_queue.put(("complete", report_count))
        
        return True
        
    except Exception as e:
        progress_queue.put(("error", str(e)))
        return False

def update_progress_ui(progress_placeholder, status_placeholder, video_placeholder):
    """Update the UI based on progress queue"""
    while not progress_queue.empty():
        message_type, content = progress_queue.get()
        
        with progress_placeholder.container():
            if message_type == "progress":
                st.progress(content)
            elif message_type == "status":
                status_placeholder.text(f"Status: {content}")
            elif message_type == "report":
                status_placeholder.success(content)
            elif message_type == "complete":
                st.progress(100)
                status_placeholder.empty()
                st.balloons()
                st.markdown(f"""
                <div class="celebration">
                    🎉 Successfully generated {content} reports! 🎉
                </div>
                """, unsafe_allow_html=True)
            elif message_type == "error":
                st.error(f"Error: {content}")




def coordinate_config_page(video_file):
    """Page for configuring tracking coordinates before processing"""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h1 style="color: var(--primary) !important;">⚙️ Configure Tracking Area</h1>
        <p>Set the detection zone and counting lines for accurate tracking</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create temp video file for preview
    temp_dir = os.path.join(REPORTS_DIR, "temp_config")
    os.makedirs(temp_dir, exist_ok=True)
    temp_video_path = os.path.join(temp_dir, "config_preview.mp4")
    
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    
    # Extract frame for reference
    cap = cv2.VideoCapture(temp_video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Could not extract frame from video")
        return None
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize default coordinates in session state
    if 'coordinates_config' not in st.session_state:
        st.session_state.coordinates_config = {
            'SOURCE': [[700, 787], [2298, 803], [5039, 2159], [-550, 2159]],
            'TARGET_WIDTH': 20,
            'TARGET_HEIGHT': 180,
            'lower_line': [[700, 787], [2298, 803]],
            'upper_line': [[150, 1606], [3750, 1606]]
        }
    
    # Display frame with current coordinates
    st.markdown("### Current Video Frame (Reference)")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame)
    ax.set_title("Configure Tracking Area")
    
    # Draw current coordinates
    poly = np.array(st.session_state.coordinates_config['SOURCE'])
    poly_patch = plt.Polygon(poly, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(poly_patch)
    
    lower_line = np.array(st.session_state.coordinates_config['lower_line'])
    upper_line = np.array(st.session_state.coordinates_config['upper_line'])
    ax.plot(lower_line[:,0], lower_line[:,1], 'g-', linewidth=2, label='Lower Line')
    ax.plot(upper_line[:,0], upper_line[:,1], 'b-', linewidth=2, label='Upper Line')
    
    ax.legend()
    st.pyplot(fig)
    
    # Coordinate configuration form
    with st.form("coordinate_config"):
        st.markdown("### Configure Coordinates")
        
        # Source polygon points
        st.markdown("#### Detection Zone (Polygon Points)")
        cols = st.columns(4)
        source_points = []
        for i in range(4):
            with cols[i]:
                st.markdown(f"**Point {i+1}**")
                x = st.number_input(f"X{i+1}", 
                                  value=st.session_state.coordinates_config['SOURCE'][i][0],
                                  key=f"source_x{i}")
                y = st.number_input(f"Y{i+1}", 
                                  value=st.session_state.coordinates_config['SOURCE'][i][1],
                                  key=f"source_y{i}")
                source_points.append([x, y])
        
        # Counting lines configuration
        st.markdown("#### Counting Lines")
        
        st.markdown("##### Lower Line (Green)")
        lower_cols = st.columns(2)
        with lower_cols[0]:
            lower_x1 = st.number_input("Lower X1", 
                                     value=st.session_state.coordinates_config['lower_line'][0][0],
                                     key="lower_x1")
            lower_y1 = st.number_input("Lower Y1", 
                                     value=st.session_state.coordinates_config['lower_line'][0][1],
                                     key="lower_y1")
        with lower_cols[1]:
            lower_x2 = st.number_input("Lower X2", 
                                     value=st.session_state.coordinates_config['lower_line'][1][0],
                                     key="lower_x2")
            lower_y2 = st.number_input("Lower Y2", 
                                     value=st.session_state.coordinates_config['lower_line'][1][1],
                                     key="lower_y2")
        
        st.markdown("##### Upper Line (Blue)")
        upper_cols = st.columns(2)
        with upper_cols[0]:
            upper_x1 = st.number_input("Upper X1", 
                                     value=st.session_state.coordinates_config['upper_line'][0][0],
                                     key="upper_x1")
            upper_y1 = st.number_input("Upper Y1", 
                                     value=st.session_state.coordinates_config['upper_line'][0][1],
                                     key="upper_y1")
        with upper_cols[1]:
            upper_x2 = st.number_input("Upper X2", 
                                     value=st.session_state.coordinates_config['upper_line'][1][0],
                                     key="upper_x2")
            upper_y2 = st.number_input("Upper Y2", 
                                     value=st.session_state.coordinates_config['upper_line'][1][1],
                                     key="upper_y2")
        
        # Target dimensions
        st.markdown("#### Target Dimensions")
        target_cols = st.columns(2)
        with target_cols[0]:
            target_width = st.number_input("Target Width", 
                                        value=st.session_state.coordinates_config['TARGET_WIDTH'],
                                        key="target_width")
        with target_cols[1]:
            target_height = st.number_input("Target Height", 
                                         value=st.session_state.coordinates_config['TARGET_HEIGHT'],
                                         key="target_height")
        
        submitted = st.form_submit_button("Update Preview")

    if submitted:
        # Update coordinates in session state
        st.session_state.coordinates_config = {
            'SOURCE': source_points,
            'TARGET_WIDTH': target_width,
            'TARGET_HEIGHT': target_height,
            'lower_line': [[lower_x1, lower_y1], [lower_x2, lower_y2]],
            'upper_line': [[upper_x1, upper_y1], [upper_x2, upper_y2]]
        }
        
        st.success("Coordinates updated! Preview below:")
        
        # Redraw with new coordinates
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame)
        ax.set_title("Configuration Preview")
        
        poly = np.array(st.session_state.coordinates_config['SOURCE'])
        poly_patch = plt.Polygon(poly, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(poly_patch)
        
        lower_line = np.array(st.session_state.coordinates_config['lower_line'])
        upper_line = np.array(st.session_state.coordinates_config['upper_line'])
        ax.plot(lower_line[:,0], lower_line[:,1], 'g-', linewidth=2, label='Lower Line')
        ax.plot(upper_line[:,0], upper_line[:,1], 'b-', linewidth=2, label='Upper Line')
        
        ax.legend()
        st.pyplot(fig)
    
    # Confirmation buttons (outside the form)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm and Process Video", key="confirm_process"):
            st.session_state['video_bytes'] = video_file.getvalue()
            st.session_state['confirmed'] = True
            st.rerun()
    with col2:
        if st.button("🔄 Reset Coordinates", key="reset_coords"):
            st.session_state.pop('coordinates_config', None)
            st.rerun()
    
    # Clean up temp files
    try:
        os.remove(temp_video_path)
    except:
        pass

# Then modify your introduction_page() function to use this new flow:
def introduction_page():

    """Render the introduction page with feature visualization"""
    # Header with gradient background
    st.markdown("""
    <div class="intro-header">
        <h1 class="intro-title">TrafficVision AI</h1>
        <p class="intro-subtitle">Smart Traffic Monitoring & Analytics System</p>
    </div>
    """, unsafe_allow_html=True)

    # Create main columns (image on left, features on right)
    col1, col2 = st.columns([1, 2])

    with col1:
        # System diagram (replace with your actual image path)
        system_img = Image.open("system.png") if os.path.exists("system.png") else None
        if system_img:
            st.image(
                system_img, 
                use_container_width=True,  # Updated parameter
                output_format="PNG"
            )

    with col2:
        # Project description
        st.markdown("""
        <div class="feature-card">
            <h3>📌 Project Description</h3>
            <p>TrafficVision AI is an advanced computer vision system that automates:</p>
            <ul>
                <li>Real-time vehicle detection and classification</li>
                <li>Precise speed measurement with violation detection</li>
                <li>Automated license plate recognition (LPR)</li>
                <li>Comprehensive PDF report generation</li>
                <li>Interactive analytics dashboard</li>
            </ul>
            <div style="margin-top: 15px;">
                <span class="tech-badge">Python</span>
                <span class="tech-badge">YOLOv11</span>
                <span class="tech-badge">PaddleOCR</span>
                <span class="tech-badge">Streamlit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards in 2 columns
        col2a, col2b = st.columns(2)
        
        with col2a:
            # Feature Card 1: Vehicle Tracking
            st.markdown("""
            <div class="feature-card">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <div style="font-size:28px;">🚗</div>
                    <h3 style="margin:0;">Vehicle Tracking</h3>
                </div>
                <p style="color:var(--text-dark);">YOLOv11 + ByteTrack for precise multi-vehicle detection and tracking across frames</p>
                <div style="display:flex; gap:5px; flex-wrap:wrap;">
                    <span class="tech-badge">YOLOv11</span>
                    <span class="tech-badge">ByteTrack</span>
                    <span class="tech-badge">90%+ Accuracy</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Card 2: Speed Analytics
            st.markdown("""
            <div class="feature-card">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <div style="font-size:28px;">📊</div>
                    <h3 style="margin:0;">Speed Analytics</h3>
                </div>
                <p style="color:var(--text-dark);">Kalman Filter + Perspective Transform for accurate speed estimation (±2km/h precision)</p>
                <div style="display:flex; gap:5px; flex-wrap:wrap;">
                    <span class="tech-badge">Kalman Filter</span>
                    <span class="tech-badge">3D Transform</span>
                    <span class="tech-badge">Real-time</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2b:
            # Feature Card 3: License Plate OCR
            st.markdown("""
            <div class="feature-card">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <div style="font-size:28px;">🔍</div>
                    <h3 style="margin:0;">License Plate Recognition</h3>
                </div>
                <p style="color:var(--text-dark);">Powered by Plate Recognizer API - Works on dark, low-res, blurry images and tough angles. Recognizes plates from 90+ countries.</p>
                <div style="display:flex; gap:5px; flex-wrap:wrap;">
                    <span class="tech-badge">API Integration</span>
                    <span class="tech-badge">90+ Countries</span>
                    <span class="tech-badge">Vehicle Type Detection</span>
                    <span class="tech-badge">Optimized for USA/India/Brazil</span>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Card 4: Automated Reports
            st.markdown("""
            <div class="feature-card">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <div style="font-size:28px;">📄</div>
                    <h3 style="margin:0;">Automated Reports</h3>
                </div>
                <p style="color:var(--text-dark);">Professional PDFs with visual evidence and violation details</p>
                <div style="display:flex; gap:5px; flex-wrap:wrap;">
                    <span class="tech-badge">FPDF</span>
                    <span class="tech-badge">Image Processing</span>
                    <span class="tech-badge">Auto-generated</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Video upload section
    st.markdown("""
    <div class="feature-card" style="margin-top:2rem;">
        <h3>🎬 Video Processing Demo</h3>
        <p>Upload traffic footage to see the system in action</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a traffic video (MP4/AVI/MOV)", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Display input video
        st.markdown("### Input Video Preview")
        st.video(uploaded_file)
        
        # Check if we're in configuration mode or ready to process
        if 'confirmed' not in st.session_state or not st.session_state.confirmed:
            coordinate_config_page(uploaded_file)
        else:
            # Show processing button only after confirmation
            if st.button("🚀 Process Video", key="process_video", use_container_width=True):
                process_video()
                st.rerun()
                # Create a temporary file for the video
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
                    tmp_video.write(st.session_state['video_file'])
                    video_path = tmp_video.name
                
                # Create a temporary config file
                with tempfile.NamedTemporaryFile(suffix=".json", mode='w', delete=False) as tmp_config:
                    json.dump(st.session_state['coordinates_config'], tmp_config)
                    config_path = tmp_config.name
                
                # Create placeholders for UI updates
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                video_placeholder = st.empty()
                
                # Start processing in a separate thread
                thread = threading.Thread(
                    target=process_video_wrapper,
                    args=(video_path, config_path, progress_placeholder, status_placeholder, video_placeholder)
                )
                
                # Add script context if needed
                if hasattr(threading.current_thread(), '_st_script_run_ctx'):
                    add_script_run_ctx(thread)
                
                thread.start()
                
                # Monitor progress
                while thread.is_alive():
                    update_progress_ui(progress_placeholder, status_placeholder, video_placeholder)
                    time.sleep(0.5)
                
                # Final update
                update_progress_ui(progress_placeholder, status_placeholder, video_placeholder)
                thread.join()
                
                # Clean up temporary files
                try:
                    os.unlink(video_path)
                    os.unlink(config_path)
                except:
                    pass
                
                # Clean up session state
                del st.session_state['confirmed']
                del st.session_state['video_file']
                del st.session_state['coordinates_config']
                st.experimental_rerun()


    # Developer profile section
    st.markdown("""
    <div class="profile-card">
        <img src="https://avatars.githubusercontent.com/u/91095645?v=4" class="profile-img">
        <h2 class="profile-title">AI & Machine Learning Engineer</h2>
        <p class="profile-subtitle">Data Scientist | Computer Vision Specialist</p>
        <div style="margin-top: 1rem;">
            <a href="https://www.linkedin.com/in/maahmoud-ibrahim/" target="_blank" class="social-link">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                </svg>
            </a>
            <a href="https://github.com/mahmoudibrahim2002" target="_blank" class="social-link">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                </svg>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
# In your app.py, modify the analysis_page function as follows:
def analysis_page():
    """Render the enhanced vehicle analysis dashboard with comprehensive insights"""
    # Header with icon
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
        <h1 style="margin: 0; color: var(--primary) !important;">🚓 Vehicle Analytics Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Video handling section - now with local FFmpeg support
    OUTPUT_VIDEO_PATH = os.path.join(REPORTS_DIR, "output.mp4")
    TEMP_VIDEO_PATH = os.path.join(REPORTS_DIR, "output_playable.mp4")
    
    if os.path.exists(OUTPUT_VIDEO_PATH):
        st.markdown("### 🎥 Processed Video Output")
        
        # Check if FFmpeg is available
        ffmpeg_available = shutil.which("ffmpeg") is not None
        
        if ffmpeg_available:
            # Re-encode the video if needed
            if not os.path.exists(TEMP_VIDEO_PATH) or \
               os.path.getmtime(OUTPUT_VIDEO_PATH) > os.path.getmtime(TEMP_VIDEO_PATH):
                try:
                    # Use ffmpeg to re-encode the video for web compatibility
                    cmd = [
                        'ffmpeg', '-y', '-i', OUTPUT_VIDEO_PATH,
                        '-c:v', 'libx264', '-preset', 'fast',
                        '-crf', '22', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart',
                        '-c:a', 'aac', '-b:a', '128k',
                        TEMP_VIDEO_PATH
                    ]
                    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    st.error(f"FFmpeg encoding failed: {e.stderr.decode('utf-8')}")
                    TEMP_VIDEO_PATH = OUTPUT_VIDEO_PATH
                except Exception as e:
                    st.error(f"Video processing error: {str(e)}")
                    TEMP_VIDEO_PATH = OUTPUT_VIDEO_PATH
        else:
            st.warning("FFmpeg not found - using original video (may not play in all browsers)")
            TEMP_VIDEO_PATH = OUTPUT_VIDEO_PATH
        
        try:
            # Display the video with controls
            with open(TEMP_VIDEO_PATH, "rb") as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
            
            # Add download button
            st.download_button(
                label="📥 Download Processed Video",
                data=video_bytes,
                file_name="processed_traffic.mp4",
                mime="video/mp4"
            )
        except Exception as e:
            st.error(f"Error displaying video: {str(e)}")
        
        st.markdown("---")
    
    # Load data with progress indicator
    with st.spinner("🔍 Loading and processing reports..."):
        df = load_data()
    
    if df.empty:
        st.warning("⚠️ No vehicle reports found in the directory")
        st.info(f"Please upload and process a video first. Reports will be saved to: {REPORTS_DIR}")
        return
    
    st.markdown("---")
    
    # Traffic Summary Section with Cards
    st.subheader("📊 Traffic Summary")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Vehicles</h3>
            <p class="metric-value">{len(df)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        violation_count = df['violation'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Violations</h3>
            <p class="metric-value">{violation_count}</p>
            <p class="metric-label">({violation_count/len(df):.1%} of total)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_speed = df['speed'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Speed</h3>
            <p class="metric-value">{avg_speed:.1f}</p>
            <p class="metric-label">km/h</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_speed = df['speed'].max()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Max Speed</h3>
            <p class="metric-value">{max_speed:.1f}</p>
            <p class="metric-label">km/h</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced Vehicle Type Analysis
    st.subheader("🚗 Vehicle Type Analysis")
    
    if 'vehicle_type' in df.columns:
        # Clean and standardize vehicle type data
        df['vehicle_type'] = df['vehicle_type'].str.title().str.strip()
        df['vehicle_type'] = df['vehicle_type'].replace({
            'Car': 'Passenger Car',
            'Truck': 'Commercial Truck',
            'Motorcycle': 'Motorcycle/Scooter',
            'Bus': 'Bus/Coach',
            '': 'Unknown',
            'Van': 'Van/Minivan'
        })
        
        # Create tabs for different visualizations
        type_tab1, type_tab2 = st.tabs(["Distribution", "Speed Analysis"])
        
        with type_tab1:
            # Vehicle type distribution
            st.markdown("#### Vehicle Type Distribution")
            
            type_counts = df['vehicle_type'].value_counts().reset_index()
            type_counts.columns = ['Vehicle Type', 'Count']
            
            fig = px.pie(
                type_counts,
                values='Count',
                names='Vehicle Type',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a bar chart for counts
            st.markdown("#### Vehicle Count by Type")
            fig = px.bar(
                type_counts,
                x='Vehicle Type',
                y='Count',
                color='Vehicle Type',
                text='Count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with type_tab2:
            # Speed analysis by vehicle type
            st.markdown("#### Speed Distribution by Vehicle Type")
            
            # Box plot for speed distribution
            fig = px.box(
                df,
                x='vehicle_type',
                y='speed',
                color='vehicle_type',
                points="all",
                title="Speed Distribution by Vehicle Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Speed percentiles by type
            st.markdown("#### Speed Statistics by Vehicle Type")
            speed_stats = df.groupby('vehicle_type')['speed'].describe()
            st.dataframe(
                speed_stats.style.background_gradient(cmap='Blues', subset=['mean', '50%', 'max']),
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Enhanced Speed Analysis Section
    st.subheader("📈 Speed Analysis")
    
    if 'speed' in df.columns:
        speed_tab1, speed_tab2 = st.tabs(["Distribution", "Violations"])
        
        with speed_tab1:
            # Enhanced speed distribution visualization
            st.markdown("#### Speed Distribution")
            
            # Interactive histogram with bin control
            bin_size = st.slider("Select bin size", 1, 20, 5, key='speed_bins')
            
            fig = px.histogram(
                df,
                x='speed',
                nbins=int((df['speed'].max() - df['speed'].min()) / bin_size),
                color_discrete_sequence=['#3a86ff'],
                marginal="rug",
                title="Speed Distribution"
            )
            fig.add_vline(x=90, line_dash="dash", line_color="red", 
                         annotation_text="Speed Limit", annotation_position="top")
            st.plotly_chart(fig, use_container_width=True)
            
            # Cumulative distribution
            st.markdown("#### Cumulative Speed Distribution")
            fig = px.ecdf(
                df,
                x='speed',
                title="Cumulative Distribution of Vehicle Speeds"
            )
            fig.add_vline(x=90, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with speed_tab2:
            # Violation analysis
            st.markdown("#### Speed Violations Analysis")
            
            # Violations by speed range
            df['speed_range'] = pd.cut(
                df['speed'],
                bins=[0, 60, 80, 90, 120, df['speed'].max()],
                labels=['0-60', '60-80', '80-90', '90-120', '120+']
            )
            
            violation_rate = df.groupby('speed_range')['violation'].mean().reset_index()
            
            fig = px.bar(
                violation_rate,
                x='speed_range',
                y='violation',
                color='violation',
                color_continuous_scale='Reds',
                labels={'violation': 'Violation Rate', 'speed_range': 'Speed Range (km/h)'},
                title="Violation Rate by Speed Range"
            )
            fig.add_hline(y=0, line_color="black")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top speed violators
            st.markdown("#### Top Speed Violators")
            violators = df[df['violation']].sort_values('speed', ascending=False)
            st.dataframe(
                violators[['tracker_id', 'vehicle_type', 'speed', 'license_plate']].head(10),
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Raw Data Section with Enhanced Filtering
    st.subheader("📋 Detailed Vehicle Reports")
    
    # Create expandable filters
    with st.expander("🔍 Filter Options", expanded=False):
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            min_speed = st.slider(
                "Minimum Speed (km/h)", 
                min_value=int(df['speed'].min()), 
                max_value=int(df['speed'].max()),
                value=int(df['speed'].min())
            )
            
            max_speed = st.slider(
                "Maximum Speed (km/h)", 
                min_value=int(df['speed'].min()), 
                max_value=int(df['speed'].max()),
                value=int(df['speed'].max())
            )
        
        with filter_col2:
            show_violations = st.selectbox(
                "Violation Status",
                options=["All", "Violations Only", "Non-Violations Only"],
                index=0
            )
            
            if 'vehicle_type' in df.columns:
                selected_types = st.multiselect(
                    "Vehicle Types",
                    options=df['vehicle_type'].unique(),
                    default=df['vehicle_type'].unique()
                )
    
    # Apply filters
    filtered_df = df[
        (df['speed'] >= min_speed) & 
        (df['speed'] <= max_speed)
    ]
    
    if show_violations == "Violations Only":
        filtered_df = filtered_df[filtered_df['violation'] == True]
    elif show_violations == "Non-Violations Only":
        filtered_df = filtered_df[filtered_df['violation'] == False]
    
    if 'vehicle_type' in df.columns and len(selected_types) > 0:
        filtered_df = filtered_df[filtered_df['vehicle_type'].isin(selected_types)]
    
    # Display filtered data with enhanced table
    default_cols = ['tracker_id', 'vehicle_type', 'license_plate', 'speed', 'violation', 'entry_time']
    available_cols = [col for col in default_cols if col in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_cols].sort_values('speed', ascending=False),
        column_config={
            "tracker_id": "Vehicle ID",
            "vehicle_type": "Type",
            "license_plate": "License Plate",
            "speed": st.column_config.NumberColumn(
                "Speed (km/h)",
                format="%.1f km/h"
            ),
            "violation": st.column_config.CheckboxColumn("Violation"),
            "entry_time": "Entry Time"
        },
        use_container_width=True,
        height=500,
        hide_index=True
    )
    
    # Export options
    st.markdown("### 📤 Export Data")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("📄 Export to CSV"):
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="vehicle_reports.csv",
                mime="text/csv"
            )
    
    with export_col2:
        if st.button("📊 Export to Excel"):
            excel_file = BytesIO()
            filtered_df.to_excel(excel_file, index=False)
            st.download_button(
                label="Download Excel",
                data=excel_file.getvalue(),
                file_name="vehicle_reports.xlsx",
                mime="application/vnd.ms-excel"
            )
    
    # Detailed Report View
    if len(filtered_df) > 0:
        st.markdown("---")
        st.subheader("📄 Individual Report Inspection")
        
        selected_report = st.selectbox(
            "Select a vehicle to view details",
            filtered_df['file_name'].unique(),
            key="report_selector"
        )
        
        report_path = os.path.join(REPORTS_DIR, selected_report)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📄 PDF View", "🔍 Text Analysis"])
        
        with tab1:
            st.markdown(f"#### Report: {selected_report}")
            display_pdf(report_path)
        
        with tab2:
            try:
                with open(report_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n".join([page.extract_text() or "" for page in reader.pages])
                
                st.markdown("#### Extracted Text Content")
                st.code(text, language='text')
                
                st.markdown("#### Key Information")
                extracted_data = extract_all_fields(text)
                st.json(extracted_data)
                
                # Add visual analysis of extracted data
                if 'speed' in extracted_data:
                    st.markdown("#### Speed Analysis")
                    speed_value = int(extracted_data['speed'])
                    
                    fig = px.bar(
                        x=['Vehicle Speed'],
                        y=[speed_value],
                        title=f"Vehicle Speed: {speed_value} km/h",
                        labels={'y': 'Speed (km/h)', 'x': ''},
                        color=[speed_value],
                        color_continuous_scale=['green', 'yellow', 'red'],
                        range_color=[0, 150]
                    )
                    fig.add_hline(y=90, line_dash="dash", line_color="red", 
                                annotation_text="Speed Limit", annotation_position="top")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not analyze report: {str(e)}")

def main():
    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <style>
            .sidebar-logo {
                width: 80%;
                margin: 0 auto;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                transition: transform 0.3s;
            }
            .sidebar-logo:hover {
                transform: scale(1.05);
            }
        </style>
        <img src="https://i.ibb.co/gX7jSgL/white.png" 
             class="sidebar-logo"
             alt="TrafficVisionAI Logo">
        <h2 style="color: white !important; margin-top: 0.5rem;">TrafficVisionAI</h2>
    </div>
    """, unsafe_allow_html=True)

    
    # Navigation buttons with icons
    nav_options = {
        "🏠 Introduction": "intro",
        "📊 Analysis Dashboard": "analysis"
    }
    
    if 'page' not in st.session_state:
        st.session_state.page = "intro"
    
    for label, page in nav_options.items():
        if st.sidebar.button(label, key=f"nav_{page}"):
            st.session_state.page = page
    
    st.sidebar.markdown("---")
    
    # System Info
    st.sidebar.markdown("""
    <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
        <p><strong>Last Processed:</strong></p>
        <p>{}</p>
        <p><strong>Reports Generated:</strong> {}</p>
    </div>
    """.format(
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        len(glob.glob(os.path.join(REPORTS_DIR, "*.pdf")))
    ), unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Footer
    st.sidebar.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <small style="color: rgba(255,255,255,0.5) !important;">Beta Version • AI Traffic System</small>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.page == "intro":
        introduction_page()
    else:
        analysis_page()

if __name__ == "__main__":
    main()