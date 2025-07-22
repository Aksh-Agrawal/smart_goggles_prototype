import os
import logging
import json
import base64
import requests
from dotenv import load_dotenv
import cv2

class SceneSummarizer:
    def __init__(self, model_type="gemini"):
        """
        Initialize the scene summarizer module
        
        Args:
            model_type (str): Type of AI model to use ('gemini' or 'openai')
        """
        load_dotenv()
        self.model_type = model_type.lower()
        
        # Initialize model-specific properties
        if self.model_type == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            self.api_available = bool(self.api_key)
            # gemini-pro-vision was deprecated on July 12, 2024, now using gemini-1.5-flash
            self.model_name = "gemini-1.5-flash"
            logging.info(f"Using Gemini model: {self.model_name}")
        elif self.model_type == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.api_available = bool(self.api_key)
            self.model_name = "gpt-4-vision-preview"
        else:
            logging.error(f"Unsupported model type: {model_type}")
            self.api_available = False
            
        if not self.api_available:
            logging.warning(f"No API key provided for {model_type}. Scene summarizer will be disabled.")
    
    def summarize_scene(self, image, prompt=None):
        """
        Summarize the current scene using a vision model
        
        Args:
            image (numpy.ndarray): Input image
            prompt (str): Custom prompt to use for the model
            
        Returns:
            str: Scene description
        """
        if not self.api_available:
            return "Scene summarizer is not available. Please configure an API key."
            
        if prompt is None:
            prompt = ("You are assisting a visually impaired person. "
                     "Describe what you see in the image clearly and concisely, "
                     "focusing on important elements, people, obstacles, and text. "
                     "Keep your description under 100 words.")
            
        try:
            if self.model_type == "gemini":
                return self._summarize_with_gemini(image, prompt)
            elif self.model_type == "openai":
                return self._summarize_with_openai(image, prompt)
            else:
                return "Unsupported model type"
        except Exception as e:
            logging.error(f"Error in scene summarization: {e}")
            return "Error getting scene description."
    
    def _summarize_with_gemini(self, image, prompt):
        """
        Summarize using Google's Gemini Vision model
        
        Args:
            image (numpy.ndarray): Input image
            prompt (str): Prompt for the model
            
        Returns:
            str: Scene description
        """
        try:
            # Debug the API key
            if not self.api_key:
                logging.error("Gemini API key is missing. Please check your .env file.")
                return "Scene description unavailable: Gemini API key is missing."
            
            logging.info(f"Using Gemini API key: {self.api_key[:5]}...{self.api_key[-4:]}")
            
            # Convert image to base64
            _, buffer = cv2.imencode(".jpg", image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            
            # Resize the image if it's too large (Gemini has size limits)
            # Target size around 1MB
            img_size_bytes = len(image_base64)
            logging.info(f"Image size before compression: {img_size_bytes/1024/1024:.2f} MB")
            
            if img_size_bytes > 1024 * 1024:  # If larger than 1MB
                # Compress the image
                img_resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                _, buffer = cv2.imencode(".jpg", img_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                image_base64 = base64.b64encode(buffer).decode("utf-8")
                logging.info(f"Image resized. New size: {len(image_base64)/1024/1024:.2f} MB")
            
            # Prepare the API request
            # Using the model name from the constructor (now gemini-1.5-flash)
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Make the API call
            logging.info("Sending request to Gemini API using model: gemini-1.5-flash")
            response = requests.post(url, headers=headers, json=data)
            
            # Check for HTTP errors
            if response.status_code != 200:
                logging.error(f"Gemini API HTTP error: {response.status_code} - {response.text}")
                return f"Scene description unavailable: API error {response.status_code}"
                
            try:
                response_data = response.json()
                logging.info(f"Gemini API response received. Status code: {response.status_code}")
                
                # Log full response for debugging (only in development)
                logging.debug(f"Full Gemini API response: {json.dumps(response_data, indent=2)}")
                
                # Extract and return the response text
                # Check for response structure in gemini-1.5-flash model
                if "candidates" in response_data and response_data["candidates"]:
                    candidate = response_data["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
                
                # If we didn't return yet, something went wrong with parsing
                error_msg = response_data.get("error", {}).get("message", "Unknown response structure")
                logging.error(f"Unexpected Gemini API response structure: {error_msg}")
                return f"Error processing the scene: Unable to parse API response"
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e}")
                return "Error processing the scene: Invalid API response format"
            except Exception as e:
                error_msg = str(e)
                logging.error(f"Unexpected error processing Gemini API response: {error_msg}")
                return f"Error processing the scene: {error_msg}"
        except Exception as e:
            logging.error(f"Error with Gemini Vision API: {e}")
            return f"Error processing the scene with Gemini Vision: {str(e)}"
    
    def _summarize_with_openai(self, image, prompt):
        """
        Summarize using OpenAI's GPT-4 Vision model
        
        Args:
            image (numpy.ndarray): Input image
            prompt (str): Prompt for the model
            
        Returns:
            str: Scene description
        """
        try:
            # Convert image to base64
            _, buffer = cv2.imencode(".jpg", image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")
            
            # Prepare the API request
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                "max_tokens": 300
            }
            
            # Make the API call
            response = requests.post(url, headers=headers, json=data)
            response_data = response.json()
            
            # Extract and return the response text
            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"]
            else:
                logging.error(f"Unexpected OpenAI API response: {response_data}")
                return "Error processing the scene."
        except Exception as e:
            logging.error(f"Error with OpenAI Vision API: {e}")
            return "Error processing the scene with OpenAI Vision."
