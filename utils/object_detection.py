import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the YOLO object detector
        
        Args:
            model_path (str): Path to the YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        
    def detect_objects(self, frame):
        """
        Detect objects in a frame using YOLOv8
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detected objects with their bounding boxes, labels, and confidence scores
            numpy.ndarray: Annotated frame with bounding boxes
        """
        results = self.model(frame)
        
        # Process the results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class and confidence
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                cls_name = self.model.names[cls_id]
                
                detection = {
                    "label": cls_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2)
                }
                detections.append(detection)
        
        # Plot the detections on the image
        annotated_frame = results[0].plot()
        
        return detections, annotated_frame
    
    def get_center_objects(self, detections, frame, threshold=0.2):
        """
        Get objects located in the center region of the frame
        
        Args:
            detections (list): List of detected objects
            frame (numpy.ndarray): Input frame
            threshold (float): Size threshold for proximity detection
            
        Returns:
            list: List of objects in the center region
            list: List of objects that are too close
        """
        center_objects = []
        proximity_objects = []
        
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Define the center region (middle 1/3 of the image)
        center_region_x1 = width // 3
        center_region_x2 = 2 * width // 3
        center_region_y1 = height // 3
        center_region_y2 = 2 * height // 3
        
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            
            # Calculate object center
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            
            # Check if object is in the center region
            if (center_region_x1 <= obj_center_x <= center_region_x2 and 
                center_region_y1 <= obj_center_y <= center_region_y2):
                center_objects.append(detection)
            
            # Calculate object size relative to frame
            obj_width = x2 - x1
            obj_height = y2 - y1
            relative_size = (obj_width * obj_height) / (width * height)
            
            # Check if object is close (large size)
            if relative_size > threshold:
                proximity_objects.append(detection)
        
        return center_objects, proximity_objects
