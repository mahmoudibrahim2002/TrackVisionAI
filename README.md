# 🚗 TrafficVision AI - Smart Traffic Monitoring & Analytics System

## 🌐 Overview
**TrafficVision AI** is an advanced computer vision system that automates vehicle tracking, speed estimation, and traffic analysis. Built with YOLOv11, ByteTrack, and License Plate Recognition, it provides comprehensive traffic monitoring with:
- Real-time vehicle detection and classification
- Precise speed measurement (±2km/h accuracy)
- Automated license plate recognition (90+ countries)
- PDF violation reports generation
- Interactive analytics dashboard

[![TrafficVision AI Demo](https://i.ibb.co/X9k3jHz/Untitled-2.png)](https://youtu.be/Clan8smF0LM) Click the thumbnail above to watch the demo on YouTube

## ✨ Key Features

### 🚦 Core Tracking
- **YOLOv11 Detection**: 90%+ accuracy vehicle detection
- **ByteTrack Tracking**: Robust multi-object tracking
- **Perspective Transform**: Accurate real-world speed estimation
- **Kalman Filtering**: Noise-resistant speed calculations

### 📊 Analytics
- Speed violation detection (configurable thresholds)
- Traffic intensity classification (Low/Moderate/Heavy)
- Directional vehicle counting (Northbound/Southbound)
- Vehicle type statistics (Cars, Trucks, Buses, etc.)
  
### 🔍 License Plate Recognition
- **Plate Recognizer API Integration**
- Works on dark, low-res (480p), and blurry images
- Handles tough angles (up to 45° skew)
- Recognizes plates from 90+ countries
- Vehicle type/model detection
- Optimized for USA/India/Brazil plates

### 📄 Automated Reporting
- Professional PDF reports with visual evidence
- License plate recognition via Plate Recognizer API
- Violation documentation with timestamps
- Batch processing for multiple videos

## 🛠️ Technical Stack

### Computer Vision
| Component          | Technology       |
|--------------------|------------------|
| Object Detection   | YOLOv11          |
| Object Tracking    | ByteTrack        |


### Backend
| Component          | Technology       |
|--------------------|------------------|
| API Integration    | Plate Recognizer |
| Report Generation  | FPDF2            |
| Data Processing    | NumPy/Pandas     |

### Frontend
| Component          | Technology       |
|--------------------|------------------|
| Dashboard          | Streamlit        |
| Visualization      | Plotly           |
| UI Styling         | Custom CSS       |

## 📦 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended)
- FFmpeg (for video processing)

### Setup
# Clone repository
!git clone https://github.com/mahmoudibrahim2002/Vehicle-Tracking-System.git
cd Vehicle-Tracking-System

# Install dependencies
pip install -r requirements.txt

# Set environment variables
"PLATE_RECOGNIZER_API_KEY=your_api_key_here" in vehicle_tracking.py

## 🚀 Usage

### Streamlit Interface
streamlit run app.py

## 📬 Contact
- 🤝 **Connect on LinkedIn**: [LinkedIn](mahmoud-ibrahim2002)

### 🙏 Acknowledgments
- **Plate Recognizer for OCR API**
