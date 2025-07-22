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
            
            # Initialize speech module
            logging.info("Initializing speech module...")
            self.speech = SpeechModule(use_gtts=self.config["use_gtts"])
            
            # Initialize face recognizer
            logging.info("Initializing face recognizer...")
            self.face_recognizer = FaceRecognizer(known_faces_dir="known_faces")
            
            # Initialize OCR module
            logging.info("Initializing OCR module...")
            self.ocr = OCRModule(tesseract_path=self.config["tesseract_path"])
            
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
            language = self.config.get("speech_language", "en-US")
                
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
        # Extract text from the frame
        text = self.ocr.extract_text(frame)
        
        # Highlight text areas in the frame
        highlighted_frame, _ = self.ocr.highlight_text_areas(frame)
        
        # If text was found, speak it
        if text and len(text) > 3:  # Ensure there's meaningful text
            self.speech.speak(f"I see text that says: {text}")
            
        return highlighted_frame
    
    def run(self):
        """Run the main application loop"""
        self.running = True
        
        # Initialize camera
        cap = setup_camera(
            camera_id=self.config["camera_id"],
            width=self.config["frame_width"],
            height=self.config["frame_height"],
            fps=self.config["fps"]
        )
        
        if not cap.isOpened():
            self.speech.speak("Failed to open camera. Exiting.")
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
                    # Reset mode after OCR
                    self.current_mode = "normal"
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
                    
                    # Alert if objects are too close, but exclude people (known or unknown)
                    if proximity_objects and not known_person_close:
                        close_object_labels = [obj["label"] for obj in proximity_objects]
                        
                        # Filter out any labels containing "person"
                        non_person_labels = [label for label in close_object_labels if "person" not in label.lower()]
                        unique_labels = list(set(non_person_labels))
                        
                        # Only issue warnings for non-person objects
                        if unique_labels:
                            if len(unique_labels) == 1:
                                self.speech.speak(f"Warning! A {unique_labels[0]} is very close to you!", priority=True)
                            else:
                                self.speech.speak("Warning! There are objects very close to you!", priority=True)
                    
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
                    self.describe_center_objects()
                elif key == ord('f'):  # Face recognition
                    self.describe_faces()
                elif key == ord('r'):  # Read text (OCR)
                    self.current_mode = "ocr"
                    self.speech.speak("Switching to OCR mode. Reading text.")
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
