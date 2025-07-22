import os
import logging
import cv2
import numpy as np
from dotenv import load_dotenv

# Configure logging
def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("smart_goggles.log"),
            logging.StreamHandler()
        ]
    )

# Initialize configuration
def load_config():
    """Load configuration from .env file"""
    load_dotenv()
    config = {
        # Object detection configuration
        "yolo_model": os.getenv("YOLO_MODEL", "yolov8n.pt"),
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
        "proximity_threshold": float(os.getenv("PROXIMITY_THRESHOLD", "0.2")),
        
        # Speech configuration
        "use_gtts": os.getenv("USE_GTTS", "False").lower() in ("true", "1", "yes"),
        "speech_language": os.getenv("SPEECH_LANGUAGE", "en-US"),
        "speech_recognition_engine": os.getenv("SPEECH_RECOGNITION_ENGINE", "google"),
        
        # OCR configuration
        "tesseract_path": os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        
        # Vision AI model configuration
        "vision_model": os.getenv("VISION_MODEL", "gemini"),  # or "openai"
        
        # Camera configuration
        "camera_id": int(os.getenv("CAMERA_ID", "0")),
        "frame_width": int(os.getenv("FRAME_WIDTH", "640")),
        "frame_height": int(os.getenv("FRAME_HEIGHT", "480")),
        "fps": int(os.getenv("FPS", "30")),
    }
    
    return config

# Initialize camera
def setup_camera(camera_id=0, width=640, height=480, fps=30):
    """
    Set up camera capture
    
    Args:
        camera_id (int): Camera device ID
        width (int): Frame width
        height (int): Frame height
        fps (int): Frames per second
        
    Returns:
        cv2.VideoCapture: Camera capture object
    """
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if not cap.isOpened():
        logging.error(f"Failed to open camera with ID {camera_id}")
    else:
        logging.info(f"Camera initialized with ID {camera_id}, resolution {width}x{height}, {fps} FPS")
        
    return cap

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ["known_faces"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")
            
# Check for required packages
def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        # Check for OpenCV
        cv2.__version__
        
        # Check for Ultralytics
        import ultralytics
        
        # Check for face_recognition
        import face_recognition
        
        # Check for speech_recognition
        import speech_recognition
        
        # Check for pyttsx3 or gTTS
        try:
            import pyttsx3
        except ImportError:
            import gtts
            
        # Check for pytesseract
        import pytesseract
        
        # Check for other useful packages
        import requests
        import threading
        import playsound
        
        logging.info("All required dependencies are installed")
        return True
    except ImportError as e:
        logging.error(f"Missing dependency: {e}")
        return False
