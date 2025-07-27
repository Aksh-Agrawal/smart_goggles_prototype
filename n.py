import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the YOLO object detector with enhanced capabilities
        
        Args:
            model_path (str): Path to the YOLOv8 model weights
        """
        self.model = YOLO(model_path)
        self.position_mapping = {
            'top_left': (0, 0, 1/3, 1/3),
            'top_center': (1/3, 0, 2/3, 1/3),
            'top_right': (2/3, 0, 1, 1/3),
            'middle_left': (0, 1/3, 1/3, 2/3),
            'center': (1/3, 1/3, 2/3, 2/3),
            'middle_right': (2/3, 1/3, 1, 2/3),
            'bottom_left': (0, 2/3, 1/3, 1),
            'bottom_center': (1/3, 2/3, 2/3, 1),
            'bottom_right': (2/3, 2/3, 1, 1)
        }
        
    def _analyze_object_features(self, frame, box):
        """
        Analyze visual features of an object region
        
        Args:
            frame (numpy.ndarray): Input frame
            box (tuple): Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            dict: Visual features of the object
        """
        try:
            x1, y1, x2, y2 = box
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Extract region of interest
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return {"has_text_like_features": False, "color_analysis": {}, "texture_score": 0}
            
            # Convert to grayscale for analysis
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Detect potential text-like regions using edge detection
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate texture features
            texture_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            
            # Color analysis
            if len(roi.shape) == 3:
                mean_color = np.mean(roi, axis=(0, 1))
                dominant_color = self._get_dominant_color(roi)
                color_analysis = {
                    "mean_bgr": mean_color.tolist(),
                    "dominant_color": dominant_color,
                    "brightness": np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                }
            else:
                color_analysis = {"brightness": np.mean(gray_roi)}
            
            # Heuristic for text-like features (high edge density + moderate texture)
            has_text_like_features = edge_density > 0.05 and 100 < texture_score < 2000
            
            return {
                "has_text_like_features": has_text_like_features,
                "edge_density": round(edge_density, 4),
                "texture_score": round(texture_score, 2),
                "color_analysis": color_analysis
            }
            
        except Exception as e:
            print(f"Feature analysis error: {e}")
            return {"has_text_like_features": False, "color_analysis": {}, "texture_score": 0}
    
    def _get_dominant_color(self, roi):
        """
        Get the dominant color in a region
        
        Args:
            roi (numpy.ndarray): Region of interest
            
        Returns:
            list: Dominant BGR color
        """
        try:
            # Reshape the image to be a list of pixels
            pixels = roi.reshape((-1, 3))
            
            # Use k-means clustering to find dominant color
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            return kmeans.cluster_centers_[0].tolist()
        except:
            # Fallback to mean color if sklearn is not available
            return np.mean(roi, axis=(0, 1)).tolist()
    
    def _get_object_position(self, box, frame_width, frame_height):
        """
        Determine the position of an object in the frame using 9-grid system
        
        Args:
            box (tuple): Bounding box coordinates (x1, y1, x2, y2)
            frame_width (int): Frame width
            frame_height (int): Frame height
            
        Returns:
            str: Position description
        """
        x1, y1, x2, y2 = box
        
        # Calculate object center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalize coordinates
        norm_x = center_x / frame_width
        norm_y = center_y / frame_height
        
        # Determine position based on 9-grid system
        for position, (min_x, min_y, max_x, max_y) in self.position_mapping.items():
            if min_x <= norm_x < max_x and min_y <= norm_y < max_y:
                return position
        
        return "unknown"
    
    def _get_detailed_position(self, box, frame_width, frame_height):
        """
        Get detailed position information including relative coordinates
        
        Args:
            box (tuple): Bounding box coordinates
            frame_width (int): Frame width  
            frame_height (int): Frame height
            
        Returns:
            dict: Detailed position information
        """
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate distances from edges
        distance_from_left = center_x / frame_width
        distance_from_top = center_y / frame_height
        distance_from_right = (frame_width - center_x) / frame_width
        distance_from_bottom = (frame_height - center_y) / frame_height
        
        return {
            "grid_position": self._get_object_position(box, frame_width, frame_height),
            "relative_x": round(distance_from_left, 3),
            "relative_y": round(distance_from_top, 3),
            "distance_from_edges": {
                "left": round(distance_from_left * 100, 1),
                "top": round(distance_from_top * 100, 1),
                "right": round(distance_from_right * 100, 1),
                "bottom": round(distance_from_bottom * 100, 1)
            }
        }
    
    def _group_similar_objects(self, detections):
        """
        Group similar objects and add instance numbers
        
        Args:
            detections (list): List of detected objects
            
        Returns:
            dict: Grouped objects with instance information
        """
        grouped_objects = defaultdict(list)
        
        for detection in detections:
            label = detection["label"]
            grouped_objects[label].append(detection)
        
        # Add instance numbers and sort by position
        for label, objects in grouped_objects.items():
            if len(objects) > 1:
                # Sort objects by their x-coordinate for consistent numbering
                objects.sort(key=lambda obj: obj["box"][0])
                for i, obj in enumerate(objects, 1):
                    obj["instance"] = i
                    obj["total_instances"] = len(objects)
            else:
                objects[0]["instance"] = 1
                objects[0]["total_instances"] = 1
        
        return grouped_objects

    def detect_objects(self, frame):
        """
        Detect objects in a frame using YOLOv8 with enhanced features
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detected objects with enhanced information
            numpy.ndarray: Annotated frame with bounding boxes
        """
        results = self.model(frame)
        frame_height, frame_width = frame.shape[:2]
        
        # Process the results
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get class and confidence
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    
                    cls_name = self.model.names[cls_id]
                    
                    # Analyze visual features for potential text-bearing objects
                    visual_features = {}
                    text_objects = ['book', 'newspaper', 'magazine', 'sign', 'poster', 'card', 'laptop', 'tv', 'monitor']
                    if any(text_obj in cls_name.lower() for text_obj in text_objects):
                        visual_features = self._analyze_object_features(frame, (x1, y1, x2, y2))
                    
                    # Get detailed position information
                    position_info = self._get_detailed_position((x1, y1, x2, y2), frame_width, frame_height)
                    
                    # Calculate object dimensions
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    relative_area = area / (frame_width * frame_height)
                    
                    detection = {
                        "label": cls_name,
                        "confidence": round(conf, 3),
                        "box": (x1, y1, x2, y2),
                        "position": position_info,
                        "dimensions": {
                            "width": width,
                            "height": height,
                            "area": area,
                            "relative_area": round(relative_area, 4)
                        },
                        "visual_features": visual_features,
                        "has_text_like_features": visual_features.get("has_text_like_features", False)
                    }
                    detections.append(detection)
        
        # Group similar objects and add instance information
        grouped_objects = self._group_similar_objects(detections)
        
        # Update detections with grouping information
        for detection in detections:
            label = detection["label"]
            for obj in grouped_objects[label]:
                if obj["box"] == detection["box"]:
                    detection["instance"] = obj["instance"]
                    detection["total_instances"] = obj["total_instances"]
                    break
        
        # Create enhanced annotated frame
        annotated_frame = self._create_enhanced_annotations(frame, detections)
        
        return detections, annotated_frame
    
    def _create_enhanced_annotations(self, frame, detections):
        """
        Create enhanced annotations with position and text information
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detected objects
            
        Returns:
            numpy.ndarray: Enhanced annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            label = detection["label"]
            conf = detection["confidence"]
            position = detection["position"]["grid_position"]
            instance = detection.get("instance", 1)
            total_instances = detection.get("total_instances", 1)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for normal objects
            if detection["has_text_like_features"]:
                color = (255, 0, 0)  # Blue for objects with text-like features
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create enhanced label
            if total_instances > 1:
                enhanced_label = f"{label} {instance}/{total_instances}"
            else:
                enhanced_label = label
            
            label_text = f"{enhanced_label} {conf:.2f} [{position}]"
            
            # Add visual feature info if available
            if detection["has_text_like_features"]:
                texture_score = detection["visual_features"].get("texture_score", 0)
                label_text += f"\nFeatures: Text-like (T:{texture_score:.0f})"
            
            # Draw label background
            label_size = cv2.getTextSize(label_text.split('\n')[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label_text.split('\n')[0], (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw additional text info if available
            if '\n' in label_text:
                cv2.putText(annotated_frame, label_text.split('\n')[1], (x1, y2+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated_frame
    
    def get_center_objects(self, detections, frame, threshold=0.2):
        """
        Get objects located in the center region of the frame with enhanced analysis
        
        Args:
            detections (list): List of detected objects
            frame (numpy.ndarray): Input frame
            threshold (float): Size threshold for proximity detection
            
        Returns:
            list: List of objects in the center region with enhanced info
            list: List of objects that are too close with enhanced info
        """
        center_objects = []
        proximity_objects = []
        
        height, width = frame.shape[:2]
        
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
            
            # Enhanced center detection with confidence scoring
            center_score = 0
            if (center_region_x1 <= obj_center_x <= center_region_x2 and 
                center_region_y1 <= obj_center_y <= center_region_y2):
                
                # Calculate how centered the object is (0-1 scale)
                center_x_norm = abs(obj_center_x - width//2) / (width//2)
                center_y_norm = abs(obj_center_y - height//2) / (height//2)
                center_score = 1 - (center_x_norm + center_y_norm) / 2
                
                enhanced_detection = detection.copy()
                enhanced_detection["center_score"] = round(center_score, 3)
                enhanced_detection["distance_from_center"] = {
                    "pixels": round(math.sqrt((obj_center_x - width//2)**2 + (obj_center_y - height//2)**2), 1),
                    "relative": round(center_score, 3)
                }
                center_objects.append(enhanced_detection)
            
            # Enhanced proximity detection
            obj_width = x2 - x1
            obj_height = y2 - y1
            relative_size = (obj_width * obj_height) / (width * height)
            
            if relative_size > threshold:
                enhanced_detection = detection.copy()
                enhanced_detection["proximity_score"] = round(relative_size, 4)
                enhanced_detection["size_category"] = self._categorize_object_size(relative_size)
                proximity_objects.append(enhanced_detection)
        
        # Sort center objects by center score (most centered first)
        center_objects.sort(key=lambda x: x["center_score"], reverse=True)
        
        # Sort proximity objects by size (largest first)
        proximity_objects.sort(key=lambda x: x["proximity_score"], reverse=True)
        
        return center_objects, proximity_objects
    
    def _categorize_object_size(self, relative_size):
        """
        Categorize object size based on relative area
        
        Args:
            relative_size (float): Relative size of object (0-1)
            
        Returns:
            str: Size category
        """
        if relative_size > 0.5:
            return "very_large"
        elif relative_size > 0.3:
            return "large"
        elif relative_size > 0.15:
            return "medium"
        elif relative_size > 0.05:
            return "small"
        else:
            return "very_small"
    
    def get_object_summary(self, detections):
        """
        Get a comprehensive summary of detected objects
        
        Args:
            detections (list): List of detected objects
            
        Returns:
            dict: Comprehensive summary of detection results
        """
        if not detections:
            return {"total_objects": 0, "summary": "No objects detected"}
        
        summary = {
            "total_objects": len(detections),
            "unique_classes": len(set(d["label"] for d in detections)),
            "objects_with_text_features": len([d for d in detections if d["has_text_like_features"]]),
            "class_distribution": {},
            "position_distribution": {},
            "confidence_stats": {
                "average": round(np.mean([d["confidence"] for d in detections]), 3),
                "highest": round(max(d["confidence"] for d in detections), 3),
                "lowest": round(min(d["confidence"] for d in detections), 3)
            },
            "detailed_objects": []
        }
        
        # Class distribution
        for detection in detections:
            label = detection["label"]
            if label not in summary["class_distribution"]:
                summary["class_distribution"][label] = {
                    "count": 0,
                    "with_text_features": 0,
                    "positions": [],
                    "instances": []
                }
            
            summary["class_distribution"][label]["count"] += 1
            if detection["has_text_like_features"]:
                summary["class_distribution"][label]["with_text_features"] += 1
            
            summary["class_distribution"][label]["positions"].append(
                detection["position"]["grid_position"]
            )
            
            # Create detailed object info
            obj_info = {
                "label": detection["label"],
                "instance": f"{detection.get('instance', 1)}/{detection.get('total_instances', 1)}",
                "position": detection["position"]["grid_position"],
                "confidence": detection["confidence"],
                "has_text_features": detection["has_text_like_features"]
            }
            
            if detection["has_text_like_features"] and detection["visual_features"]:
                obj_info["visual_analysis"] = {
                    "texture_score": detection["visual_features"].get("texture_score", 0),
                    "edge_density": detection["visual_features"].get("edge_density", 0),
                    "brightness": detection["visual_features"].get("color_analysis", {}).get("brightness", 0)
                }
            
            summary["detailed_objects"].append(obj_info)
        
        # Position distribution
        for detection in detections:
            pos = detection["position"]["grid_position"]
            summary["position_distribution"][pos] = summary["position_distribution"].get(pos, 0) + 1
        
        return summary