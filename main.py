import os
import cv2
import time
import logging
import threading
import numpy as np
from datetime import datetime

# Import utility modules
from utils import (
    ObjectDetector,
    SpeechModule,
    FaceRecognizer,
    OCRModule,
    EmergencySystem,
    SceneSummarizer,
    setup_logging,
    load_config,
    setup_camera,
    ensure_directories,
    check_dependencies
)

class SmartGoggles:
    def __init__(self):
        """Initialize the Smart Goggles application"""
        # Set up logging
        setup_logging()
        
        # Check dependencies
        if not check_dependencies():
            logging.error("Missing dependencies. Please install required packages.")
            return
            
        # Create necessary directories
        ensure_directories()
        
        # Load configuration
        self.config = load_config()
        
        # Initialize modules
        self.init_modules()
        
        # Initialize state variables
        self.running = False
        self.current_mode = "normal"  # Modes: normal, ocr, navigation
        self.last_command_time = 0
        self.command_cooldown = 2  # seconds between voice commands
        self.listen_thread = None
        self.listen_active = False
        
    def init_modules(self):
        """Initialize all the modules"""
        try:
            # Initialize object detector
            logging.info("Initializing object detector...")
            self.object_detector = ObjectDetector(model_path=self.config["yolo_model"])
            
            # Initialize speech module with Indian English preference
            logging.info("Initializing speech module...")
            self.speech = SpeechModule(use_gtts=self.config["use_gtts"], language="en-IN")
            
            # Initialize face recognizer
            logging.info("Initializing face recognizer...")
            self.face_recognizer = FaceRecognizer(known_faces_dir="known_faces")
            
            # Initialize OCR module
            logging.info("Initializing OCR module...")
            self.ocr = OCRModule(tesseract_path=self.config["tesseract_path"], api_key=self.config["ocr_api_key"])
            
            # Initialize emergency system
            logging.info("Initializing emergency system...")
            self.emergency = EmergencySystem()
            
            # Initialize scene summarizer
            logging.info("Initializing scene summarizer...")
            self.scene_summarizer = SceneSummarizer(model_type=self.config["vision_model"])
            
            logging.info("All modules initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing modules: {e}")
            raise
    
    def start_voice_recognition(self):
        """Start the voice command recognition thread"""
        if self.listen_thread is None or not self.listen_thread.is_alive():
            self.listen_active = True
            self.listen_thread = threading.Thread(target=self._voice_command_loop)
            self.listen_thread.daemon = True
            self.listen_thread.start()
    
    def _voice_command_loop(self):
        """Background thread for voice command recognition"""
        consecutive_failures = 0
        max_failures = 3  # After this many failures, provide feedback
        
        while self.running and self.listen_active:
            # Don't listen while speaking
            if self.speech.is_speaking:
                time.sleep(0.5)
                continue
                
            # Check cooldown period
            if time.time() - self.last_command_time < self.command_cooldown:
                time.sleep(0.5)
                continue
            
            # Get language from config if available
            language = self.config.get("speech_language", "en-IN")
                
            # Listen for command with enhanced parameters
            command = self.speech.listen(
                timeout=4,  # Longer timeout for better detection
                phrase_time_limit=6,  # Longer phrase time limit
                language=language
            )
            
            if command:
                logging.info(f"Recognized command: {command}")
                self.last_command_time = time.time()
                self.process_voice_command(command)
                consecutive_failures = 0  # Reset failure counter on success
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    # After multiple failed attempts, provide feedback
                    self.speech.speak("I didn't hear you clearly. Please try speaking again.", priority=False)
                    consecutive_failures = 0  # Reset counter
                    self.last_command_time = time.time()  # Prevent immediate retry
                
            # Small delay to prevent CPU overuse
            time.sleep(0.1)
    
    def process_voice_command(self, command):
        """
        Process voice commands
        
        Args:
            command (str): Recognized voice command
        """
        command = command.lower()
        
        # Check for object detection command
        if "something" in command and "front" in command:
            self.describe_center_objects()
            
        # Check for face recognition mode
        elif "who" in command and ("there" in command or "front" in command):
            self.describe_faces()
            
        # Check for OCR mode
        elif "read" in command and "front" in command:
            self.current_mode = "ocr"
            self.speech.speak("Switching to OCR mode. Reading text.")
            self.ocr_mode_start_time = time.time()
            if hasattr(self, 'no_text_spoken'):
                delattr(self, 'no_text_spoken')
            if hasattr(self, 'last_spoken_text'):
                delattr(self, 'last_spoken_text')
            
        # Check for scene description command
        elif "what's happening" in command or "what is happening" in command or "around me" in command:
            self.describe_scene()
            
        # Check for emergency command
        elif "help" in command and "me" in command:
            self.trigger_emergency()
            
        # Check for exit command
        elif "exit" in command or "quit" in command or "stop" in command:
            self.speech.speak("Stopping Smart Goggles.")
            self.running = False
    
    def describe_center_objects(self):
        """Describe objects in the center of the frame"""
        if not hasattr(self, 'current_center_objects') or not self.current_center_objects:
            self.speech.speak("I don't see anything in front of you right now.")
            return
            
        objects = [obj["label"] for obj in self.current_center_objects]
        if objects:
            if len(objects) == 1:
                self.speech.speak(f"I see a {objects[0]} in front of you.")
            else:
                object_str = ", ".join(objects[:-1]) + " and " + objects[-1]
                self.speech.speak(f"I see {object_str} in front of you.")
        else:
            self.speech.speak("I don't see anything specific in front of you right now.")
    
    def describe_faces(self):
        """Describe detected faces"""
        if not hasattr(self, 'current_faces') or not self.current_faces:
            self.speech.speak("I don't see any people in front of you right now.")
            return
            
        if len(self.current_faces) == 1:
            name, _ = self.current_faces[0]
            if name == "Unknown":
                self.speech.speak("There is someone in front of you, but I don't recognize them.")
            else:
                self.speech.speak(f"I can see {name} in front of you.")
        else:
            # Only count unknown people
            unknown_count = sum(1 for name, _ in self.current_faces if name == "Unknown")
            
            # Only announce if there are unknown people
            if unknown_count > 0:
                if unknown_count == 1:
                    self.speech.speak("There is one person I don't recognize.")
                else:
                    self.speech.speak(f"There are {unknown_count} people I don't recognize.")
            # Stay silent if only known people are detected
    
    def describe_scene(self):
        """Describe the current scene using AI vision model"""
        if not hasattr(self, 'current_frame'):
            self.speech.speak("I can't describe the scene right now.")
            return
            
        self.speech.speak("Analyzing the scene around you. This will take a moment.")
        
        # Copy the current frame to avoid modification during processing
        frame_copy = self.current_frame.copy()
        
        # Debug - log attempt to describe scene
        logging.info("Attempting to describe scene with model type: " + self.config["vision_model"])
        
        # Call the scene summarizer
        description = self.scene_summarizer.summarize_scene(frame_copy)
        
        # Log the description for debugging
        logging.info(f"Scene description result: {description[:100]}...")
        
        # If it's an error message, display additional debugging info
        if "error" in description.lower() or "unavailable" in description.lower():
            logging.error(f"Scene description failed: {description}")
            api_key = self.scene_summarizer.api_key
            if api_key:
                masked_key = f"{api_key[:5]}...{api_key[-4:]}"
                logging.info(f"API key being used: {masked_key}")
            else:
                logging.error("No API key available")
        
        # Speak the description
        self.speech.speak(description)
    
    def trigger_emergency(self):
        """Trigger emergency mode with alerts"""
        self.speech.speak("Emergency mode activated. Sending alerts!", priority=True)
        
        # Play emergency sound right away in a separate thread to avoid blocking
        import threading
        sound_thread = threading.Thread(target=self.emergency.play_emergency_sound)
        sound_thread.daemon = True
        sound_thread.start()
        
        # Set a default location (would be GPS coordinates in a real implementation)
        location = "Unknown location"
        
        # Create a simplified emergency message
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"EMERGENCY ALERT! Smart Goggles user needs help! Time: {time_str}"
        
        # Display visible alert on screen if possible
        try:
            emergency_frame = np.zeros((300, 600, 3), dtype=np.uint8)
            emergency_frame[:] = (0, 0, 255)  # Red background
            
            # Add alert text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(emergency_frame, "EMERGENCY ALERT", (100, 100), font, 1.2, (255, 255, 255), 2)
            cv2.putText(emergency_frame, "Sending SMS alert...", (100, 150), font, 0.7, (255, 255, 255), 2)
            
            # Show in a separate window
            cv2.imshow("EMERGENCY ALERT", emergency_frame)
            cv2.waitKey(1)  # Update the window
        except Exception as e:
            logging.error(f"Could not display emergency visual alert: {e}")
        
        # Send the SMS alert with direct method
        success = self.emergency.send_emergency_alert(message, location)
        
        # Provide immediate feedback
        if success:
            self.speech.speak("Emergency alert has been sent.", priority=True)
        else:
            # Try one more time with a simplified message
            try:
                from twilio.rest import Client
                import socket
                import os
                from dotenv import load_dotenv
                
                # Ensure environment variables are loaded
                load_dotenv()
                
                # Get credentials from environment variables
                account_sid = os.getenv("TWILIO_ACCOUNT_SID")
                auth_token = os.getenv("TWILIO_AUTH_TOKEN")
                from_number = os.getenv("TWILIO_FROM_NUMBER")
                to_number = os.getenv("TWILIO_TO_NUMBER")
                
                # Use a very short timeout
                socket.setdefaulttimeout(3)
                
                # Only attempt to send if we have valid credentials
                if account_sid and auth_token and from_number and to_number:
                    # Send simplified message
                    client = Client(account_sid, auth_token)
                    client.messages.create(
                        from_=from_number,
                        body='EMERGENCY! Need help now!',
                        to=to_number
                    )
                self.speech.speak("Alert sent successfully.", priority=True)
            except:
                # If all fails, just play the sound
                self.emergency.play_emergency_sound()
                self.speech.speak("Could not send alert. Using local alarm only.", priority=True)
    
    # Removed the backup method since we've simplified the emergency process
    
    def process_ocr(self, frame):
        """
        Process OCR on the current frame
        
        Args:
            frame (numpy.ndarray): Current frame
            
        Returns:
            numpy.ndarray: Frame with highlighted text regions
        """
        try:
            # Store the frame time to avoid repeated OCR on same frame
            current_time = time.time()
            
            # More frequent OCR processing (every 1 second instead of 3)
            if hasattr(self, 'last_ocr_time') and current_time - self.last_ocr_time < 1:
                # Just return the previously highlighted frame if available
                if hasattr(self, 'last_highlighted_frame'):
                    return self.last_highlighted_frame
                return frame
                
            self.last_ocr_time = current_time
            
            # Log OCR processing
            logging.info("Processing OCR on current frame")
            
            # Prepare the frame for better text detection
            # 1. Create a copy for processing
            ocr_frame = frame.copy()
            
            # 2. Optional: Adjust contrast and brightness for better text visibility
            alpha = 1.3  # Contrast control (1.0-3.0)
            beta = 10    # Brightness control (0-100)
            adjusted_frame = cv2.convertScaleAbs(ocr_frame, alpha=alpha, beta=beta)
            
            # Extract text from the enhanced frame with advanced processing
            text = self.ocr.extract_text(adjusted_frame, preprocess=True, use_enhanced_ocr=True)
            
            # Highlight text areas in the frame with confidence visualization
            # Lower threshold to catch more potential text (40 instead of 50)
            highlighted_frame, text_boxes = self.ocr.highlight_text_areas(frame, confidence_threshold=40, show_confidence=True)
            
            # Create a visual indicator that OCR mode is active
            cv2.rectangle(highlighted_frame, (0, 0), (highlighted_frame.shape[1], 40), (0, 0, 200), -1)
            cv2.putText(highlighted_frame, "OCR MODE ACTIVE", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Store the highlighted frame
            self.last_highlighted_frame = highlighted_frame
            
            # Log the extracted text
            logging.info(f"OCR extracted text: '{text}'")
            
            # If text was found, speak it
            if text and len(text) > 2:  # Even short texts can be important (e.g., "GO", "NO")
                # Avoid speaking the same text repeatedly
                if not hasattr(self, 'last_spoken_text') or text != self.last_spoken_text:
                    # Visualize the detected text more prominently
                    text_display = highlighted_frame.copy()
                    
                    # Create a semi-transparent text box
                    overlay = highlighted_frame.copy()
                    cv2.rectangle(overlay, (0, highlighted_frame.shape[0] - 60), 
                                 (highlighted_frame.shape[1], highlighted_frame.shape[0]), 
                                 (0, 0, 0), -1)
                    
                    # Blend the overlay with the original frame
                    cv2.addWeighted(overlay, 0.7, highlighted_frame, 0.3, 0, highlighted_frame)
                    
                    # Add text overlay to the frame with larger, more visible font
                    cv2.putText(highlighted_frame, "TEXT: " + text[:50], 
                                (10, highlighted_frame.shape[0] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # Speak the detected text
                    self.speech.speak(f"I see text that says: {text}")
                    self.last_spoken_text = text
            else:
                # If no text found after 1.5 seconds, provide feedback
                if hasattr(self, 'ocr_mode_start_time') and current_time - self.ocr_mode_start_time > 1.5:
                    if not hasattr(self, 'no_text_spoken') or not self.no_text_spoken:
                        # Add a visual indicator that no text was found
                        overlay = highlighted_frame.copy()
                        cv2.rectangle(overlay, (0, highlighted_frame.shape[0] - 60), 
                                     (highlighted_frame.shape[1], highlighted_frame.shape[0]), 
                                     (0, 0, 0), -1)
                        
                        # Blend the overlay with the original frame
                        cv2.addWeighted(overlay, 0.7, highlighted_frame, 0.3, 0, highlighted_frame)
                        
                        # Add text overlay
                        cv2.putText(highlighted_frame, "No text detected. Try adjusting camera.", 
                                    (10, highlighted_frame.shape[0] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                        
                        self.speech.speak("No readable text detected. Try adjusting the camera position.")
                        self.no_text_spoken = True
                        
                # If still no text after a longer period, provide another hint
                if hasattr(self, 'ocr_mode_start_time') and current_time - self.ocr_mode_start_time > 4:
                    if not hasattr(self, 'second_hint_given') or not self.second_hint_given:
                        self.speech.speak("Make sure text is well-lit and centered in view.")
                        self.second_hint_given = True
                        
            return highlighted_frame
            
        except Exception as e:
            logging.error(f"Error in OCR processing: {e}")
            return frame
    
    def run(self):
        """Run the main application loop"""
        self.running = True
        
        # First try to initialize camera with ID 1 (external camera)
        logging.info("Trying to initialize camera with ID 1...")
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])
        cap.set(cv2.CAP_PROP_FPS, self.config["fps"])
        
        # If camera 1 is not available, fall back to camera 0
        if not cap.isOpened():
            logging.info("Camera ID 1 not available, falling back to camera ID 0...")
            cap.release()  # Release the failed capture
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])
            cap.set(cv2.CAP_PROP_FPS, self.config["fps"])
            
        if not cap.isOpened():
            self.speech.speak("Failed to open any camera. Exiting.")
            return
            
            # Welcome message
        self.speech.speak("Smart Goggles activated. Use keyboard commands to control the system.")
        
        # Display initial help to guide the user
        self.show_help()
        
        try:
            # Main processing loop
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    logging.error("Failed to grab frame from camera")
                    break
                    
                # Store current frame for other functions to use
                self.current_frame = frame.copy()
                
                # Process frame based on current mode
                if self.current_mode == "ocr":
                    # Process frame with OCR
                    display_frame = self.process_ocr(frame)
                    # Don't reset mode immediately - allow for multiple frames of OCR processing
                    # The mode will be reset after 5 seconds
                    if not hasattr(self, 'ocr_mode_start_time'):
                        self.ocr_mode_start_time = time.time()
                    elif time.time() - self.ocr_mode_start_time > 5:  # 5 second OCR mode
                        self.current_mode = "normal"
                        self.speech.speak("Exiting OCR mode.")
                        if hasattr(self, 'ocr_mode_start_time'):
                            delattr(self, 'ocr_mode_start_time')
                else:
                    # Run object detection
                    detections, annotated_frame = self.object_detector.detect_objects(frame)
                    display_frame = annotated_frame
                    
                    # Run face recognition first to identify if people are present
                    face_results = self.face_recognizer.recognize_faces(display_frame)
                    
                    # Get objects in the center and check proximity
                    center_objects, proximity_objects = self.object_detector.get_center_objects(
                        detections, frame, threshold=self.config["proximity_threshold"]
                    )
                    
                    # Store current center objects for voice commands
                    self.current_center_objects = center_objects
                    
                    # Check if a known person is among the close objects
                    known_person_close = False
                    if face_results:
                        for name, (left, top, right, bottom) in face_results:
                            if name != "Unknown":
                                # Check if this known person is close (large face)
                                face_width = right - left
                                face_height = bottom - top
                                face_size = (face_width * face_height) / (frame.shape[1] * frame.shape[0])
                                if face_size > self.config["proximity_threshold"]:
                                    known_person_close = True
                                    break
                    
                  
                  
                    # Store current faces for voice commands
                    self.current_faces = face_results
                
                # Add command instructions to the frame
                display_frame = self.add_command_overlay(display_frame)
                
                # Display the frame
                cv2.imshow("Smart Goggles", display_frame)
                
                # Process keyboard commands
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    break
                elif key == ord('o'):  # Object detection
                    if self.current_mode == "ocr":
                        self.current_mode = "normal"
                        self.speech.speak("Exiting OCR mode.")
                        if hasattr(self, 'ocr_mode_start_time'):
                            delattr(self, 'ocr_mode_start_time')
                    self.describe_center_objects()
                elif key == ord('f'):  # Face recognition
                    if self.current_mode == "ocr":
                        self.current_mode = "normal"
                        self.speech.speak("Exiting OCR mode.")
                        if hasattr(self, 'ocr_mode_start_time'):
                            delattr(self, 'ocr_mode_start_time')
                    self.describe_faces()
                elif key == ord('r'):  # Read text (OCR)
                    self.current_mode = "ocr"
                    self.speech.speak("Switching to OCR mode. Reading text.")
                    self.ocr_mode_start_time = time.time()
                    if hasattr(self, 'no_text_spoken'):
                        delattr(self, 'no_text_spoken')
                    if hasattr(self, 'last_spoken_text'):
                        delattr(self, 'last_spoken_text')
                elif key == ord('s'):  # Scene description
                    self.describe_scene()
                elif key == ord('e'):  # Emergency
                    self.trigger_emergency()
                elif key == ord('h'):  # Help/Instructions
                    self.show_help()
                    
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            self.speech.speak("An error occurred. Smart Goggles is shutting down.")
        finally:
            # Clean up resources
            self.running = False
            if self.listen_thread and self.listen_thread.is_alive():
                self.listen_active = False
                self.listen_thread.join(timeout=1)
                
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Smart Goggles terminated")
            
    def add_command_overlay(self, frame):
        """
        Add keyboard command instructions overlay to the frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with command overlay
        """
        # Create a copy of the frame to avoid modifying the original
        display_frame = frame.copy()
        
        # Define command keys and their functions
        commands = [
            "O: Detect Objects",
            "F: Identify Faces",
            "R: Read Text (OCR)",
            "S: Describe Scene",
            "E: Emergency Alert",
            "H: Show Help",
            "Q: Quit"
        ]
        
        # Settings for the overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background
        padding = 5
        
        # Add semi-transparent background for better visibility
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (200, len(commands) * 20 + 10), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, display_frame)
        
        # Add the command instructions
        for i, cmd in enumerate(commands):
            y_pos = 20 * (i + 1)
            cv2.putText(display_frame, cmd, (10, y_pos), font, font_scale, text_color, font_thickness)
        
        return display_frame
    
    def show_help(self):
        """Display help information about keyboard commands"""
        help_text = """
        Smart Goggles Keyboard Commands:
        
        O - Detect and describe objects in front of you
        F - Identify faces in view
        R - Read text in view (OCR mode)
        S - Describe the entire scene
        E - Trigger emergency mode
        H - Show this help information
        Q - Quit the application
        
        Press any key to close this help screen.
        """
        
        # Create a black image for help screen
        height, width = 400, 600
        help_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Settings for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 1
        text_color = (255, 255, 255)  # White
        
        # Split help text into lines and display
        y_pos = 40
        for line in help_text.strip().split('\n'):
            cv2.putText(help_image, line, (20, y_pos), font, font_scale, text_color, font_thickness)
            y_pos += 30
        
        # Display help window
        cv2.imshow("Smart Goggles - Help", help_image)
        self.speech.speak("Displaying keyboard command help. Press any key to close.")
        
        # Wait for any key press
        cv2.waitKey(0)
        cv2.destroyWindow("Smart Goggles - Help")

if __name__ == "__main__":
    app = SmartGoggles()
    app.run()
