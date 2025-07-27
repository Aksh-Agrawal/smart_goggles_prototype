import cv2
import pytesseract
import numpy as np
import logging
import base64
import io
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import os
import requests
import json

class OCRModule:
    def __init__(self, tesseract_path=None, api_key=None):
        """
        Initialize the OCR module
        
        Args:
            tesseract_path (str): Path to tesseract executable (needed on Windows)
            api_key (str): Optional API key for enhanced OCR (supports various services)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # Initialize API for enhanced OCR if key is provided
        self.api_key = api_key if api_key else os.getenv("OCR_API_KEY")
        self.ai_service = os.getenv("OCR_AI_SERVICE", "azure").lower()  # default to Azure
        self.ai_endpoint = os.getenv("OCR_AI_ENDPOINT", "")
        self.enhanced_ocr_available = False
        
        if self.api_key:
            try:
                # Check which service to use based on configuration
                if self.ai_service == "azure":
                    # Azure OCR service setup - just validate key is present
                    self.enhanced_ocr_available = True
                    logging.info("Azure Computer Vision OCR initialized")
                elif self.ai_service == "aws":
                    # AWS Rekognition setup - just validate key is present
                    self.enhanced_ocr_available = True
                    logging.info("AWS Rekognition OCR initialized")
                else:
                    # Generic OCR API setup
                    self.enhanced_ocr_available = True
                    logging.info(f"{self.ai_service.capitalize()} OCR service initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize enhanced OCR API: {e}")
                
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            
    def preprocess_image(self, image, method='adaptive'):
        """
        Enhanced image preprocessing for better OCR results
        
        Args:
            image (numpy.ndarray): Input image
            method (str): Preprocessing method - 'adaptive', 'otsu', 'gaussian', 'morphology'
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply different preprocessing methods
            if method == 'adaptive':
                # Adaptive thresholding - better for varying lighting conditions
                processed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            elif method == 'gaussian':
                # Gaussian blur + OTSU threshold
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            elif method == 'morphology':
                # Morphological operations to clean up noise
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            else:  # default 'otsu'
                # Original OTSU method (improved)
                processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Additional noise reduction
            processed = cv2.medianBlur(processed, 3)
            
            # Optional: Dilation to make text thicker and more readable
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            processed = cv2.dilate(processed, kernel, iterations=1)
            
            return processed
            
        except Exception as e:
            logging.error(f"Error in image preprocessing: {e}")
            # Fallback to original method
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            blur = cv2.medianBlur(gray, 3)
            return blur
        
    def extract_text(self, image, preprocess=True, use_enhanced_ocr=True, tesseract_config='--psm 6'):
        """
        Enhanced text extraction with multiple methods and fallback options
        
        Args:
            image (numpy.ndarray): Input image
            preprocess (bool): Whether to preprocess the image
            use_enhanced_ocr (bool): Use enhanced OCR services as fallback if Tesseract confidence is low
            tesseract_config (str): Tesseract configuration parameters
            
        Returns:
            str: Extracted text
        """
        try:
            original_image = image.copy()
            best_text = ""
            best_confidence = 0
            
            # Try multiple preprocessing methods with Tesseract
            preprocessing_methods = ['adaptive', 'otsu', 'gaussian'] if preprocess else [None]
            
            for method in preprocessing_methods:
                try:
                    if method:
                        processed_image = self.preprocess_image(original_image, method)
                    else:
                        processed_image = original_image
                    
                    # Extract text with confidence scores
                    data = pytesseract.image_to_data(processed_image, config=tesseract_config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0
                    
                    text = pytesseract.image_to_string(processed_image, config=tesseract_config).strip()
                    
                    if avg_confidence > best_confidence and text:
                        best_confidence = avg_confidence
                        best_text = text
                        
                except Exception as e:
                    logging.warning(f"Tesseract method {method} failed: {e}")
                    continue
            
            # Use enhanced OCR as fallback if confidence is low or no text found
            if (best_confidence < 70 or not best_text) and use_enhanced_ocr and self.enhanced_ocr_available:
                try:
                    enhanced_text = self._extract_text_with_enhanced_ocr(original_image)
                    if enhanced_text and len(enhanced_text) > len(best_text):
                        best_text = enhanced_text
                        logging.info(f"Using enhanced OCR result from {self.ai_service}")
                except Exception as e:
                    logging.warning(f"Enhanced OCR fallback failed: {e}")
            
            # Post-process the text
            best_text = self._post_process_text(best_text)
            
            logging.info(f"OCR completed with confidence: {best_confidence:.2f}")
            return best_text
            
        except Exception as e:
            logging.error(f"Error in OCR text extraction: {e}")
            return ""
    
    def _extract_text_with_enhanced_ocr(self, image):
        """
        Extract text using enhanced OCR services (Azure, AWS, or other configured service)
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            str: Extracted text
        """
        try:
            # Convert image to appropriate format for API
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Create a buffer to store the image
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Use the appropriate OCR service based on configuration
            if self.ai_service == "azure":
                return self._extract_text_with_azure(image_bytes)
            elif self.ai_service == "aws":
                return self._extract_text_with_aws(image_bytes)
            else:
                # Generic OCR API call
                return self._extract_text_with_generic_api(image_b64)
            
        except Exception as e:
            logging.error(f"Enhanced OCR extraction failed: {e}")
            return ""
            
    def _extract_text_with_azure(self, image_bytes):
        """
        Extract text using Azure Computer Vision API
        
        Args:
            image_bytes (bytes): Image bytes
            
        Returns:
            str: Extracted text
        """
        try:
            # Placeholder for Azure Computer Vision implementation
            # In a real implementation, this would use the Azure SDK or REST API
            
            # Example with requests (simplified):
            if not self.ai_endpoint:
                self.ai_endpoint = "https://api.cognitive.microsoft.com/vision/v3.2/read/analyze"
                
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Content-Type': 'application/octet-stream'
            }
            
            # This is a simplified implementation - Azure actually uses an asynchronous model
            # that requires submitting the job and then polling for results
            response = requests.post(self.ai_endpoint, headers=headers, data=image_bytes)
            
            if response.status_code == 202:
                logging.info("Azure OCR request accepted - would normally poll for results")
                # In a real implementation: store operation-location header and poll until completion
                return "Azure OCR processing (simulated)"
            else:
                logging.warning(f"Azure OCR request failed: {response.status_code}")
                return ""
                
        except Exception as e:
            logging.error(f"Azure OCR extraction failed: {e}")
            return ""
            
    def _extract_text_with_aws(self, image_bytes):
        """
        Extract text using AWS Rekognition
        
        Args:
            image_bytes (bytes): Image bytes
            
        Returns:
            str: Extracted text
        """
        try:
            # Placeholder for AWS Rekognition implementation
            # In a real implementation, this would use the boto3 library
            
            # Mock response for simulation
            logging.info("AWS Rekognition OCR would be called here")
            return "AWS OCR processing (simulated)"
            
        except Exception as e:
            logging.error(f"AWS OCR extraction failed: {e}")
            return ""
            
    def _extract_text_with_generic_api(self, image_b64):
        """
        Extract text using a generic OCR API
        
        Args:
            image_b64 (str): Base64-encoded image
            
        Returns:
            str: Extracted text
        """
        try:
            # Placeholder for a generic OCR API implementation
            logging.info(f"Would call {self.ai_service} OCR API here")
            return "Generic OCR processing (simulated)"
            
        except Exception as e:
            logging.error(f"Generic OCR extraction failed: {e}")
            return ""
    
    def _post_process_text(self, text):
        """
        Post-process extracted text to improve quality
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return text
            
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Fix common OCR errors
        replacements = {
            '0': 'O',  # Only in specific contexts
            '1': 'l',  # Only in specific contexts
            '5': 'S',  # Only in specific contexts
            '@': 'a',
            '€': 'e',
            '£': 'L',
        }
        
        # Apply replacements only when it makes sense contextually
        # This is a simplified version - you can expand this logic
        for old, new in replacements.items():
            if old in text and not any(char.isdigit() for char in text.split()):
                text = text.replace(old, new)
        
        return text.strip()
            
    def highlight_text_areas(self, image, confidence_threshold=60, show_confidence=False):
        """
        Enhanced function to highlight areas containing text in the image
        
        Args:
            image (numpy.ndarray): Input image
            confidence_threshold (int): Minimum confidence for text detection
            show_confidence (bool): Whether to display confidence scores on boxes
            
        Returns:
            numpy.ndarray: Image with highlighted text areas
            list: List of bounding boxes for text regions with confidence scores
        """
        try:
            # Get bounding boxes for text regions with multiple PSM modes
            psm_modes = [6, 3, 8]  # Try different page segmentation modes
            best_result = None
            best_confidence = 0
            
            for psm in psm_modes:
                try:
                    config = f'--psm {psm}'
                    d = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in d['conf'] if int(conf) > 0]
                    avg_conf = np.mean(confidences) if confidences else 0
                    
                    if avg_conf > best_confidence:
                        best_confidence = avg_conf
                        best_result = d
                        
                except Exception:
                    continue
            
            if not best_result:
                return image, []
            
            d = best_result
            n_boxes = len(d['level'])
            
            # Create a copy of the image
            highlighted_image = image.copy()
            
            boxes = []
            
            # Draw boxes around text with different colors based on confidence
            for i in range(n_boxes):
                conf = int(d['conf'][i])
                if conf > confidence_threshold:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    
                    # Skip very small boxes
                    if w < 10 or h < 10:
                        continue
                    
                    # Color based on confidence: red (low) to green (high)
                    if conf >= 80:
                        color = (0, 255, 0)  # Green for high confidence
                    elif conf >= 70:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for lower confidence
                    
                    cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Optionally show confidence scores
                    if show_confidence:
                        cv2.putText(highlighted_image, f'{conf}%', (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    boxes.append((x, y, x + w, y + h, conf))
            
            logging.info(f"Found {len(boxes)} text regions with confidence > {confidence_threshold}")
            return highlighted_image, boxes
            
        except Exception as e:
            logging.error(f"Error in highlighting text areas: {e}")
            return image, []
    
    def get_text_with_coordinates(self, image):
        """
        Get text along with their coordinates for advanced processing
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of dictionaries containing text, coordinates, and confidence
        """
        try:
            d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            n_boxes = len(d['level'])
            
            results = []
            
            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:
                    text = d['text'][i].strip()
                    if text:  # Only include non-empty text
                        result = {
                            'text': text,
                            'left': d['left'][i],
                            'top': d['top'][i],
                            'width': d['width'][i],
                            'height': d['height'][i],
                            'confidence': int(d['conf'][i])
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            logging.error(f"Error getting text with coordinates: {e}")
            return []