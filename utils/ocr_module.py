import cv2
import pytesseract
import numpy as np
import logging

class OCRModule:
    def __init__(self, tesseract_path=None):
        """
        Initialize the OCR module
        
        Args:
            tesseract_path (str): Path to tesseract executable (needed on Windows)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
    def preprocess_image(self, image):
        """
        Preprocess image for better OCR results
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Apply median blur to remove noise
        blur = cv2.medianBlur(gray, 3)
        
        return blur
        
    def extract_text(self, image, preprocess=True):
        """
        Extract text from image using OCR
        
        Args:
            image (numpy.ndarray): Input image
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            str: Extracted text
        """
        try:
            if preprocess:
                image = self.preprocess_image(image)
                
            # Extract text from image
            text = pytesseract.image_to_string(image)
            
            # Clean up the text
            text = text.strip()
            
            return text
        except Exception as e:
            logging.error(f"Error in OCR text extraction: {e}")
            return ""
            
    def highlight_text_areas(self, image):
        """
        Highlight areas containing text in the image
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Image with highlighted text areas
            list: List of bounding boxes for text regions
        """
        try:
            # Get bounding boxes for text regions
            d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            n_boxes = len(d['level'])
            
            # Create a copy of the image
            highlighted_image = image.copy()
            
            boxes = []
            
            # Draw boxes around text
            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:  # Only consider high confidence detections
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    boxes.append((x, y, x + w, y + h))
            
            return highlighted_image, boxes
        except Exception as e:
            logging.error(f"Error in highlighting text areas: {e}")
            return image, []
