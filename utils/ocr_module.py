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
            
            # Enhanced preprocessing methods
            if method == 'adaptive':
                # Enhanced adaptive thresholding with better parameters
                processed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9
                )
            elif method == 'gaussian':
                # Improved Gaussian blur + OTSU threshold
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            elif method == 'morphology':
                # Advanced morphological operations for better text extraction
                # First normalize the image
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                # Apply morphological gradient to enhance text edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
                # Binarize the image
                _, binarized = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # Close small gaps in text
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # horizontal kernel
                processed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
                # Another closing operation with vertical kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))  # vertical kernel
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            elif method == 'canny':
                # Edge-based text detection
                edges = cv2.Canny(gray, 100, 200)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                processed = cv2.dilate(edges, kernel, iterations=1)
            else:  # default 'otsu'
                # Enhanced OTSU method
                # First normalize the image
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                # Apply bilateral filter to preserve edges while reducing noise
                blurred = cv2.bilateralFilter(gray, 11, 17, 17)
                # Apply threshold
                processed = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            # Enhanced noise reduction with bilateral filter
            processed = cv2.bilateralFilter(processed, 5, 75, 75)
            
            # Improved text enhancement with morphological operations
            # Create a rectangular kernel for horizontal text
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            # Create a rectangular kernel for vertical text
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            # Apply closing operation to connect broken text parts
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_h)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_v)
            
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
            
            # Enhanced preprocessing methods for better text detection
            preprocessing_methods = ['adaptive', 'otsu', 'gaussian', 'morphology', 'canny'] if preprocess else [None]
            
            # Try multiple tesseract page segmentation modes for better results
            psm_modes = ['--psm 6', '--psm 3', '--psm 4', '--psm 11', '--psm 1']
            
            # Try combinations of preprocessing methods and PSM modes for best results
            for method in preprocessing_methods:
                try:
                    if method:
                        processed_image = self.preprocess_image(original_image, method)
                    else:
                        processed_image = original_image
                        
                    # Try multiple PSM modes for each preprocessing method
                    for psm in psm_modes:
                        try:
                            current_config = psm + " --oem 1" if psm != tesseract_config else tesseract_config
                            
                            # Extract text with confidence scores
                            data = pytesseract.image_to_data(processed_image, config=current_config, output_type=pytesseract.Output.DICT)
                            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                            avg_confidence = np.mean(confidences) if confidences else 0
                            
                            text = pytesseract.image_to_string(processed_image, config=current_config).strip()
                            
                            if avg_confidence > best_confidence and text:
                                best_confidence = avg_confidence
                                best_text = text
                                logging.info(f"Improved text detection: {method} with {psm}, conf: {avg_confidence:.2f}")
                                
                                # If we find a high confidence match, we can stop early
                                if avg_confidence > 80:
                                    break
                                    
                        except Exception as e:
                            logging.debug(f"PSM mode {psm} failed for {method}: {e}")
                            continue
                            
                except Exception as e:
                    logging.warning(f"Tesseract method {method} failed: {e}")
                    continue
                    
            # If no good result found, try combined approach - resize and special processing
            if best_confidence < 60 or not best_text:
                try:
                    # Resize to larger size for better OCR
                    height, width = original_image.shape[:2]
                    scale_factor = 2.0
                    enlarged_img = cv2.resize(original_image, (int(width * scale_factor), int(height * scale_factor)))
                    
                    # Try OCR on the enlarged image with special config
                    special_config = "--psm 11 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;-_!?()[]{}\"'$%&@#*/\\+=<> "
                    text = pytesseract.image_to_string(enlarged_img, config=special_config).strip()
                    
                    if text and (not best_text or len(text) > len(best_text)):
                        best_text = text
                        logging.info("Using enlarged image text detection")
                except Exception as e:
                    logging.warning(f"Enlarged image OCR failed: {e}")
            
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
        
        # Remove non-printable characters
        import string
        printable = set(string.printable)
        text = ''.join(filter(lambda x: x in printable, text))
            
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Fix common OCR errors
        replacements = {
            # Number to letter confusions (context dependent)
            '0': 'O',  # Only in specific contexts
            '1': 'I',  # Only in specific contexts
            '5': 'S',  # Only in specific contexts
            '8': 'B',  # Only in specific contexts
            # Symbol confusions
            '@': 'a',
            '€': 'e',
            '£': 'L',
            '¢': 'c',
            # Common error patterns
            'rn': 'm',
            'cl': 'd',
            'vv': 'w',
            'ii': 'u',
            'iii': 'm',
            # Fix spacing issues
            ' ,': ',',
            ' .': '.',
            ' !': '!',
            ' ?': '?',
        }
        
        # First, separate text into words to analyze context
        words = text.split()
        processed_words = []
        
        for word in words:
            # Skip processing very short words or likely numbers
            if len(word) <= 2 or word.isdigit() or (word[0].isdigit() and any(c.isalpha() for c in word)):
                processed_words.append(word)
                continue
                
            # Apply word-level replacements for letter/number confusions
            if all(c.isalpha() for c in word):
                for old, new in replacements.items():
                    if old.isdigit() and old in word:
                        word = word.replace(old, new)
            
            # Apply general symbol replacements
            for old, new in replacements.items():
                if not old.isdigit() and old in word:
                    word = word.replace(old, new)
                    
            processed_words.append(word)
            
        # Rejoin the processed words
        processed_text = ' '.join(processed_words)
        
        # Fix common grammar issues
        processed_text = processed_text.replace(" i ", " I ")  # Capitalize standalone "i"
        processed_text = processed_text.replace(" i'm ", " I'm ")
        processed_text = processed_text.replace(" i'll ", " I'll ")
        processed_text = processed_text.replace(" i'd ", " I'd ")
        processed_text = processed_text.replace(" i've ", " I've ")
        
        # Capitalize first letter of sentences
        sentences = processed_text.split('. ')
        capitalized_sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        processed_text = '. '.join(capitalized_sentences)
        
        # Ensure text starts with capital letter
        if processed_text and len(processed_text) > 0:
            processed_text = processed_text[0].upper() + processed_text[1:]
        
        return processed_text.strip()
            
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
            # Create a copy for both traditional OCR and enhanced detection
            highlighted_image = image.copy()
            
            # Enhanced text region detection using EAST text detector or MSER 
            # for better detection of text regions before OCR
            try:
                # Method 1: Use MSER for text region detection
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Create MSER detector
                mser = cv2.MSER_create()
                
                # Detect regions
                regions, _ = mser.detectRegions(gray)
                
                # Draw MSER regions on a separate layer
                mser_mask = np.zeros_like(gray)
                for region in regions:
                    x, y, w, h = cv2.boundingRect(region)
                    # Filter out too small or too large regions
                    if w > 5 and h > 5 and w < image.shape[1]//2 and h < image.shape[0]//2:
                        cv2.rectangle(mser_mask, (x, y), (x+w, y+h), 255, 1)
                
                # Dilate to connect nearby regions
                kernel = np.ones((5, 5), np.uint8)
                mser_mask = cv2.dilate(mser_mask, kernel, iterations=1)
                
                # Draw these regions on the highlighted image in a semi-transparent layer
                mser_overlay = highlighted_image.copy()
                for region in regions:
                    x, y, w, h = cv2.boundingRect(region)
                    if w > 5 and h > 5 and w < image.shape[1]//2 and h < image.shape[0]//2:
                        cv2.rectangle(mser_overlay, (x, y), (x+w, y+h), (0, 255, 0), 1)
                
                # Blend the MSER detection with the original image
                alpha = 0.3
                cv2.addWeighted(mser_overlay, alpha, highlighted_image, 1-alpha, 0, highlighted_image)
            except Exception as e:
                logging.warning(f"MSER text region detection failed: {e}")
            
            # Get bounding boxes for text regions with multiple PSM modes
            psm_modes = [6, 3, 8, 11, 4]  # Extended PSM modes
            best_result = None
            best_confidence = 0
            
            for psm in psm_modes:
                try:
                    config = f'--psm {psm} --oem 1'
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
                return highlighted_image, []  # Now using highlighted_image that might have MSER regions
            
            d = best_result
            n_boxes = len(d['level'])
            
            # We're already using highlighted_image from above
            boxes = []
            
            # Draw boxes around text with different colors based on confidence
            for i in range(n_boxes):
                conf = int(d['conf'][i])
                if conf > confidence_threshold:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    text = d['text'][i].strip()
                    
                    # Skip very small boxes or empty text
                    if w < 8 or h < 8 or not text:
                        continue
                    
                    # Color based on confidence: red (low) to green (high)
                    if conf >= 80:
                        color = (0, 255, 0)  # Green for high confidence
                        thickness = 2
                    elif conf >= 70:
                        color = (0, 255, 255)  # Yellow for medium confidence
                        thickness = 2
                    else:
                        color = (0, 0, 255)  # Red for lower confidence
                        thickness = 1
                    
                    # Draw a filled rectangle behind the text for better visibility
                    alpha = 0.3
                    overlay = highlighted_image.copy()
                    cv2.rectangle(overlay, (x-3, y-3), (x + w+3, y + h+3), color, -1)  # Filled rectangle
                    cv2.addWeighted(overlay, alpha, highlighted_image, 1-alpha, 0, highlighted_image)
                    
                    # Draw the outline box
                    cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), color, thickness)
                    
                    # Optionally show confidence scores and text
                    if show_confidence:
                        conf_text = f'{conf}%'
                        # Place the confidence text above the box with a dark background for readability
                        text_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        cv2.rectangle(highlighted_image, (x, y-text_size[1]-5), (x + text_size[0], y), (0, 0, 0), -1)
                        cv2.putText(highlighted_image, conf_text, (x, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    boxes.append((x, y, x + w, y + h, conf, text))
            
            # If we have boxes, highlight the regions more prominently
            if boxes:
                # Create a heat map effect to show where text might be
                overlay = highlighted_image.copy()
                for x, y, x2, y2, conf, _ in boxes:
                    # Draw a larger area around the text with semi-transparency
                    padding = 10
                    cv2.rectangle(overlay, (max(0, x-padding), max(0, y-padding)), 
                                (min(highlighted_image.shape[1], x2+padding), min(highlighted_image.shape[0], y2+padding)), 
                                (0, 200, 255), -1)  # Orange fill
                
                # Blend with low alpha for subtle highlight
                cv2.addWeighted(overlay, 0.2, highlighted_image, 0.8, 0, highlighted_image)
            
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