# 🚗 TrafficVision AI - Intelligent Vehicle Tracking System

![Project Banner](https://i.ibb.co/gX7jSgL/white.png)

## 🌐 Overview
**TrafficVision AI** is an advanced computer vision system that automates vehicle tracking, speed estimation, and traffic analysis. Built with YOLOv11, ByteTrack, and PaddleOCR, it provides comprehensive traffic monitoring with:
- Real-time vehicle detection and classification
- Precise speed measurement (±2km/h accuracy)
- Automated license plate recognition (90+ countries)
- PDF violation reports generation
- Interactive analytics dashboard

[![Demo Video](https://img.youtube.com/vi/7iGKksFZZzY/maxresdefault.jpg)](https://youtu.be/7iGKksFZZzY)

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
| OCR                | PaddleOCR        |
| Video Processing   | OpenCV           |

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
git clone https://github.com/mahmoudibrahim2002/Vehicle-Tracking-System.git
cd Vehicle-Tracking-System

# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "PLATE_RECOGNIZER_API_KEY=your_api_key_here" > .env

## 🚀 Usage

### Streamlit Interface
streamlit run app.py

## 📬 Contact
- 🤝 **Connect on LinkedIn**: [LinkedIn](mahmoud-ibrahim2002)

### 🙏 Acknowledgments
- **Plate Recognizer for OCR API**
