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
        Analyze visual features of an object region with enhanced text detection
        
        Args:
            frame (numpy.ndarray): Input frame
            box (tuple): Bounding box coordinates (x1, y1, x2, y2)
            
        Returns:
            dict: Visual features of the object with text detection results
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
            
            # Advanced text-feature detection
            text_features = self._detect_text_features(gray_roi)
            
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
            
            # Combine basic detection with advanced text features
            has_text_like_features = text_features["text_confidence"] > 0.6
            
            result = {
                "has_text_like_features": has_text_like_features,
                "edge_density": round(text_features["edge_density"], 4),
                "texture_score": round(text_features["texture_score"], 2),
                "text_confidence": round(text_features["text_confidence"], 2),
                "text_pattern_score": round(text_features["text_pattern_score"], 2),
                "color_analysis": color_analysis
            }
            
            # If text is detected, add text-specific features
            if has_text_like_features:
                result["text_features"] = {
                    "horizontal_lines": text_features["horizontal_lines"],
                    "vertical_spacing": text_features["vertical_spacing"],
                    "contrast_variance": text_features["contrast_variance"]
                }
                
            return result
            
        except Exception as e:
            print(f"Feature analysis error: {e}")
            return {"has_text_like_features": False, "color_analysis": {}, "texture_score": 0}
            
    def _detect_text_features(self, gray_image):
        """
        Advanced detection of text features in grayscale image
        
        Args:
            gray_image (numpy.ndarray): Grayscale image
            
        Returns:
            dict: Text feature metrics
        """
        result = {
            "edge_density": 0,
            "texture_score": 0,
            "text_confidence": 0,
            "text_pattern_score": 0,
            "horizontal_lines": 0,
            "vertical_spacing": 0,
            "contrast_variance": 0
        }
        
        try:
            if gray_image.size == 0:
                return result
                
            # Basic edge and texture metrics
            edges = cv2.Canny(gray_image, 50, 150)
            result["edge_density"] = np.sum(edges > 0) / edges.size
            result["texture_score"] = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            
            # Calculate horizontal line patterns (text typically has many horizontal lines)
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            detected_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel_h)
            thresh_lines = cv2.threshold(detected_lines, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            line_contours, _ = cv2.findContours(thresh_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result["horizontal_lines"] = len(line_contours)
            
            # Calculate vertical spacing patterns
            y_projection = np.sum(thresh_lines, axis=1)
            zero_runs = np.where(np.diff(np.hstack(([0], (y_projection > 0).astype(int), [0]))) != 0)[0]
            spacing_diffs = np.diff(zero_runs[::2])  # Get distances between text lines
            result["vertical_spacing"] = np.std(spacing_diffs) if len(spacing_diffs) > 1 else 0
            
            # Calculate local contrast variance (text typically has high local contrast)
            local_std = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3).var()
            result["contrast_variance"] = local_std
            
            # Calculate combined text confidence score
            # Weight the features based on importance for text detection
            edge_weight = 0.3
            texture_weight = 0.2
            lines_weight = 0.25
            spacing_weight = 0.15
            contrast_weight = 0.1
            
            # Normalize each feature to 0-1 range for better combination
            norm_edge = min(1.0, result["edge_density"] / 0.1)  # Edge density above 0.1 is max
            norm_texture = min(1.0, result["texture_score"] / 1000)  # Texture score above 1000 is max
            norm_lines = min(1.0, result["horizontal_lines"] / 10)  # More than 10 lines is max
            norm_spacing = min(1.0, result["vertical_spacing"] / 5)  # Spacing deviation above 5 is max
            norm_contrast = min(1.0, result["contrast_variance"] / 1000)  # Contrast variance above 1000 is max
            
            # Calculate text confidence score
            result["text_confidence"] = (edge_weight * norm_edge + 
                                        texture_weight * norm_texture + 
                                        lines_weight * norm_lines + 
                                        spacing_weight * norm_spacing + 
                                        contrast_weight * norm_contrast)
                                        
            # Calculate text pattern score (check for uniform patterns typical of text)
            horizontal_edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            vertical_edges = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_ratio = np.sum(np.abs(horizontal_edges)) / (np.sum(np.abs(vertical_edges)) + 1e-6)
            result["text_pattern_score"] = min(1.0, max(0, (gradient_ratio - 0.5) / 1.5))
                
            return result
            
        except Exception as e:
            print(f"Text feature detection error: {e}")
            return result
            
        except Exception as e:
            print(f"Feature analysis error: {e}")
            return {"has_text_like_features": False, "color_analysis": {}, "texture_score": 0}
    
    def _get_dominant_color(self, roi):
        """
        Get the dominant color in a region using histogram analysis
        
        Args:
            roi (numpy.ndarray): Region of interest
            
        Returns:
            list: Dominant BGR color
        """
        try:
            # Method 1: Use histogram to find most frequent color ranges
            hist_b = cv2.calcHist([roi], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([roi], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([roi], [2], None, [256], [0, 256])
            
            # Find peaks in each channel
            dominant_b = np.argmax(hist_b)
            dominant_g = np.argmax(hist_g)
            dominant_r = np.argmax(hist_r)
            
            return [float(dominant_b), float(dominant_g), float(dominant_r)]
        except:
            # Fallback to mean color
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

    def detect_objects(self, frame, detect_text=False):
        """
        Detect objects in a frame using YOLOv8 with enhanced features and optional text detection
        
        Args:
            frame (numpy.ndarray): Input frame
            detect_text (bool): Whether to detect and analyze text in all objects
            
        Returns:
            list: List of detected objects with enhanced information
            numpy.ndarray: Annotated frame with bounding boxes
        """
        # Reset text detection state
        self.last_text_detection = None
        
        results = self.model(frame)
        frame_height, frame_width = frame.shape[:2]
        
        # Process the results
        detections = []
        # Track potential text regions for further analysis
        potential_text_regions = []
        
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
                    
                    # Define which objects are likely to contain text
                    text_objects = ['book', 'newspaper', 'magazine', 'sign', 'poster', 'card', 
                                   'laptop', 'tv', 'monitor', 'billboard', 'traffic sign', 'door', 
                                   'whiteboard', 'screen', 'phone', 'paper', 'document']
                    
                    # Analyze visual features for potential text-bearing objects
                    visual_features = {}
                    should_analyze = detect_text or any(text_obj in cls_name.lower() for text_obj in text_objects)
                    
                    if should_analyze:
                        visual_features = self._analyze_object_features(frame, (x1, y1, x2, y2))
                        
                        # If object has text-like features, add to potential text regions for further processing
                        if visual_features.get("text_confidence", 0) > 0.6:
                            potential_text_regions.append({
                                "box": (x1, y1, x2, y2),
                                "confidence": visual_features["text_confidence"],
                                "label": cls_name
                            })
                            # Set the flag that we've detected text
                            self.last_text_detection = True
                    
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
        
        # Create enhanced annotated frame with potential text regions
        annotated_frame = self._create_enhanced_annotations(frame, detections, potential_text_regions)
        
        # If in text detection mode, highlight potential text regions more prominently
        if detect_text and potential_text_regions:
            # Overlay with semi-transparent regions to highlight potential text
            text_overlay = annotated_frame.copy()
            
            for region in potential_text_regions:
                x1, y1, x2, y2 = region["box"]
                # Draw a filled rectangle with transparency
                cv2.rectangle(text_overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)  # Cyan fill
            
            # Blend the text overlay with the annotated frame
            cv2.addWeighted(text_overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        
        return detections, annotated_frame
    
    def _create_enhanced_annotations(self, frame, detections, potential_text_regions=None):
        """
        Create enhanced annotations with position and text information
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detected objects
            potential_text_regions (list, optional): Specifically identified text regions
            
        Returns:
            numpy.ndarray: Enhanced annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw a text detection indicator at the top if needed
        if hasattr(self, 'last_text_detection') and self.last_text_detection:
            cv2.rectangle(annotated_frame, (10, 10), (150, 40), (255, 200, 0), -1)
            cv2.putText(annotated_frame, "Text Detected", (15, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            label = detection["label"]
            conf = detection["confidence"]
            position = detection["position"]["grid_position"]
            instance = detection.get("instance", 1)
            total_instances = detection.get("total_instances", 1)
            
            # Determine appropriate color based on object features
            has_text = detection.get("has_text_like_features", False)
            
            if has_text:
                # Use a distinctive color for objects containing text (cyan)
                color = (255, 255, 0)  # Yellow for objects with text-like features
                
                # For text-containing objects, highlight potential text areas within
                if "visual_features" in detection:
                    text_confidence = detection["visual_features"].get("text_confidence", 0)
                    if text_confidence > 0.7:
                        # Draw a special indicator for high-confidence text areas
                        text_region_color = (0, 255, 255)  # Cyan for high-confidence text regions
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), text_region_color, 3)
                        
                        # Add a "TEXT" indicator next to the object
                        cv2.putText(annotated_frame, "TEXT", (x2 + 5, y1 + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_region_color, 2)
            else:
                color = (0, 255, 0)  # Green for normal objects
            
            # Draw standard bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create enhanced label
            if total_instances > 1:
                enhanced_label = f"{label} {instance}/{total_instances}"
            else:
                enhanced_label = label
            
            label_text = f"{enhanced_label} {conf:.2f} [{position}]"
            
            # Add visual feature info if available
            if has_text:
                text_conf = detection["visual_features"].get("text_confidence", 0)
                label_text += f" [Text:{text_conf:.2f}]"
            
            # Draw label background - make it more visible
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1-30), (x1 + label_size[0] + 5, y1), color, -1)
            
            # Draw label text with better visibility
            cv2.putText(annotated_frame, label_text, (x1 + 2, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Black outline
            cv2.putText(annotated_frame, label_text, (x1 + 2, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) # White text
            
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
    
    def detect_text_regions(self, frame):
        """
        Specially detect regions in the image that are likely to contain text
        for more effective OCR processing
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of regions likely containing text
            numpy.ndarray: Annotated frame with text regions highlighted
        """
        # Create a copy for annotations
        annotated_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        
        text_regions = []
        
        # Method 1: MSER detector (Maximally Stable Extremal Regions)
        try:
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            # Filter and merge regions
            for region in regions:
                x, y, w, h = cv2.boundingRect(region)
                # Filter out too small or too large regions
                if (w > 10 and h > 5 and w < frame.shape[1]//2 and h < frame.shape[0]//2 and 
                    w / h < 10 and h / w < 10):  # Reasonable aspect ratio
                    
                    # Extract ROI
                    roi = gray[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue
                        
                    # Check if ROI has text-like features
                    std_dev = np.std(roi)
                    mean_val = np.mean(roi)
                    edge_detector = cv2.Canny(roi, 50, 150)
                    edge_density = np.sum(edge_detector > 0) / edge_detector.size
                    
                    # Text regions typically have good contrast and edge density
                    if std_dev > 20 and edge_density > 0.05:
                        text_regions.append({
                            "box": (x, y, x+w, y+h),
                            "confidence": min(100, int(edge_density * 1000)),
                            "area": w * h,
                            "properties": {
                                "std_dev": std_dev,
                                "edge_density": edge_density,
                                "mean_intensity": mean_val
                            }
                        })
                        
                        # Draw box on annotated frame
                        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        except Exception as e:
            print(f"MSER detection error: {e}")
            
        # Method 2: Image gradient and morphology
        try:
            # Create structuring elements of different orientations for morphological filtering
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))  # horizontal
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13))  # vertical
            
            # Apply morphological operations to detect text components
            # For horizontal text
            grad_x = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_h)
            thresh_x = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # For vertical text
            grad_y = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_v)
            thresh_y = cv2.threshold(grad_y, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Combine horizontal and vertical components
            combined = cv2.bitwise_or(thresh_x, thresh_y)
            
            # Connect nearby text components
            kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_connect)
            
            # Find contours
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio
                if (w > 30 and h > 8 and w < frame.shape[1]*0.8 and h < frame.shape[0]*0.8 and 
                    w / h < 15 and h / w < 5):  # Reasonable text aspect ratio
                    
                    # Check for text-like properties
                    roi = gray[y:y+h, x:x+w]
                    if roi.size > 0:
                        # Calculate properties
                        std_dev = np.std(roi)
                        mean_val = np.mean(roi)
                        
                        # Add if it has sufficient contrast
                        if std_dev > 25:
                            # Check if this region overlaps with existing ones
                            new_region = True
                            for region in text_regions:
                                rx1, ry1, rx2, ry2 = region["box"]
                                overlap = max(0, min(rx2, x+w) - max(rx1, x)) * max(0, min(ry2, y+h) - max(ry1, y))
                                if overlap / (w*h) > 0.5:  # Over 50% overlap
                                    new_region = False
                                    break
                                    
                            if new_region:
                                text_regions.append({
                                    "box": (x, y, x+w, y+h),
                                    "confidence": min(90, int(std_dev)),
                                    "area": w * h,
                                    "properties": {
                                        "std_dev": std_dev,
                                        "mean_intensity": mean_val,
                                        "method": "gradient"
                                    }
                                })
                                
                                # Draw box on annotated frame with different color
                                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        except Exception as e:
            print(f"Gradient detection error: {e}")
            
        # Method 3: Detect objects likely to contain text
        detections, _ = self.detect_objects(frame)
        for detection in detections:
            if detection["has_text_like_features"]:
                x1, y1, x2, y2 = detection["box"]
                
                # Check if region overlaps with existing regions
                new_region = True
                for region in text_regions:
                    rx1, ry1, rx2, ry2 = region["box"]
                    overlap = max(0, min(rx2, x2) - max(rx1, x1)) * max(0, min(ry2, y2) - max(ry1, y1))
                    if overlap / ((x2-x1)*(y2-y1)) > 0.5:  # Over 50% overlap
                        new_region = False
                        break
                        
                if new_region:
                    text_regions.append({
                        "box": (x1, y1, x2, y2),
                        "confidence": int(detection["visual_features"].get("texture_score", 50)),
                        "area": (x2-x1)*(y2-y1),
                        "properties": {
                            "object_type": detection["label"],
                            "confidence": detection["confidence"]
                        }
                    })
                    
                    # Draw box on annotated frame with different color
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Sort regions by confidence (highest first)
        text_regions.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Add labels to annotated frame
        for i, region in enumerate(text_regions):
            x1, y1, x2, y2 = region["box"]
            conf = region["confidence"]
            cv2.putText(annotated_frame, f"Text {i+1} ({conf}%)", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return text_regions, annotated_frame
    
    def extract_text_from_regions(self, frame, ocr_module):
        """
        Extract text from detected text regions in the frame using the OCR module
        with enhanced preprocessing and visualization
        
        Args:
            frame (numpy.ndarray): Input frame
            ocr_module: OCR module instance
            
        Returns:
            dict: Extracted text with regions and confidence scores
            numpy.ndarray: Annotated frame with text and regions
        """
        # Get text regions
        text_regions, annotated_frame = self.detect_text_regions(frame)
        
        results = {
            "regions": [],
            "full_text": ""
        }
        
        all_text = []
        confidence_sum = 0
        
        # Create a copy for visualization
        visual_frame = annotated_frame.copy()
        
        # Process each region with OCR using multiple enhancement methods
        for i, region in enumerate(text_regions):
            x1, y1, x2, y2 = region["box"]
            
            # Extract region with a small margin for better context
            margin = 5
            x1_m = max(0, x1 - margin)
            y1_m = max(0, y1 - margin)
            x2_m = min(frame.shape[1], x2 + margin)
            y2_m = min(frame.shape[0], y2 + margin)
            
            roi = frame[y1_m:y2_m, x1_m:x2_m]
            if roi.size == 0:
                continue
                
            # Try multiple preprocessing approaches for better results
            text_results = []
            
            # Method 1: Standard preprocessing
            text1 = ocr_module.extract_text(roi, preprocess=True, use_enhanced_ocr=True)
            if text1 and len(text1) > 1:
                text_results.append((text1, 70))  # Base confidence
            
            # Method 2: Enhanced contrast
            alpha = 1.5  # Increased contrast
            beta = 15    # Increased brightness
            enhanced_roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)
            text2 = ocr_module.extract_text(enhanced_roi, preprocess=True, use_enhanced_ocr=True)
            
            # Only add if different from previous result and not empty
            if text2 and len(text2) > 1 and text2 != text1:
                text_results.append((text2, 60))
            
            # Method 3: Adaptive thresholding for difficult text
            try:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
                text3 = ocr_module.extract_text(thresh_roi, preprocess=False, use_enhanced_ocr=True)
                
                # Only add if different and not empty
                if text3 and len(text3) > 1 and text3 not in [text1, text2]:
                    text_results.append((text3, 50))
            except Exception as e:
                print(f"Thresholding error: {e}")
            
            # Select best result based on length and content quality
            if text_results:
                # Prioritize by length and character quality
                best_text, best_confidence = max(text_results, key=lambda x: len(x[0]))
                
                # Check if the text seems valid (has reasonable character distribution)
                invalid_chars = sum(1 for c in best_text if not (c.isalnum() or c.isspace() or c in '.,!?:;-_'))
                valid_ratio = 1 - (invalid_chars / len(best_text) if len(best_text) > 0 else 0)
                
                # Adjust confidence based on text quality
                adjusted_confidence = int(best_confidence * valid_ratio)
                
                results["regions"].append({
                    "region_id": i+1,
                    "box": (x1, y1, x2, y2),
                    "text": best_text,
                    "confidence": adjusted_confidence
                })
                all_text.append(best_text)
                confidence_sum += adjusted_confidence
                
                # Add text to annotated frame with visual indication of confidence
                confidence_color = (0, min(255, adjusted_confidence * 2.5), 255)
                cv2.rectangle(visual_frame, (x1, y1), (x2, y2), confidence_color, 2)
                
                # Draw text with background for readability
                text_to_display = best_text[:20] + "..." if len(best_text) > 20 else best_text
                
                # Create text background with alpha transparency
                overlay = visual_frame.copy()
                text_bg_height = 30
                cv2.rectangle(overlay, (x1, y2), (x1 + 300, y2 + text_bg_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, visual_frame, 0.3, 0, visual_frame)
                
                # Draw text with confidence indicator
                cv2.putText(visual_frame, f"{text_to_display} ({adjusted_confidence}%)", (x1 + 5, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Combine all text
        results["full_text"] = " ".join(all_text)
        
        # Calculate average confidence if there are regions
        if results["regions"]:
            results["avg_confidence"] = confidence_sum / len(results["regions"])
        else:
            results["avg_confidence"] = 0
        
        # Add summary to the visual frame with more details
        if all_text:
            # Create a semi-transparent header
            header_h = 50
            overlay = visual_frame.copy()
            cv2.rectangle(overlay, (0, 0), (visual_frame.shape[1], header_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, visual_frame, 0.3, 0, visual_frame)
            
            # Add detailed summary
            avg_conf = results["avg_confidence"]
            conf_color = (0, min(255, avg_conf * 2.5), 255)
            summary_text = f"Found {len(results['regions'])} text regions - Avg confidence: {avg_conf:.1f}%"
            cv2.putText(visual_frame, summary_text, (15, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
        
        return results, visual_frame