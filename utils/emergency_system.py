import logging
import os
import time
import socket
from dotenv import load_dotenv

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logging.warning("Twilio not installed. SMS alerts will not be available.")

class EmergencySystem:
    def __init__(self):
        """
        Initialize the emergency alert system
        """
        load_dotenv()
        
        # Set up detailed logging
        logging.info("Initializing emergency alert system...")
        
        # Load Twilio credentials from environment variables
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.to_number = os.getenv("TWILIO_TO_NUMBER")
        
        # For compatibility with existing code
        self.email_available = False
        self.sms_available = TWILIO_AVAILABLE
        
        logging.info("Emergency system initialized with direct Twilio method")
    
    def send_emergency_alert(self, message, location="Unknown location"):
        """
        Send emergency alert via SMS only
        
        Args:
            message (str): Alert message
            location (str): Current location
        
        Returns:
            bool: True if alert was sent, False otherwise
        """
        full_message = f"{message}\nLocation: {location}"
        logging.info("Sending emergency SMS alert...")
        
        # Send SMS directly with minimal overhead
        return self._send_direct_sms(full_message)
    
    def _send_direct_sms(self, message):
        """
        Send SMS alert directly using hardcoded Twilio credentials
        This optimized method is designed for speed and reliability
        
        Args:
            message (str): Alert message to send
            
        Returns:
            bool: True if SMS was sent successfully, False otherwise
        """
        if not TWILIO_AVAILABLE:
            logging.error("Twilio package not installed - cannot send SMS")
            return False
            
        try:
            logging.info("Sending SMS using optimized direct method...")
            
            # Set a short timeout to prevent hanging
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(3)  # 3-second timeout for faster response
            
            # Initialize client with hardcoded credentials
            client = Client(self.account_sid, self.auth_token)
            
            # Send the message
            message_obj = client.messages.create(
                from_=self.from_number,
                body=message,
                to=self.to_number
            )
            
            # Reset socket timeout
            socket.setdefaulttimeout(old_timeout)
            
            logging.info(f"SMS sent successfully, SID: {message_obj.sid}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send SMS: {e}")
            return False
  

    def play_emergency_sound(self):
        """
        Play emergency sound
        """
        # Print ASCII bell character multiple times for a loud alarm
        logging.info("Playing emergency sound...")
        
        try:
            # First try to use winsound if available (Windows only)
            try:
                import winsound
                # Beep at 750 Hz for 500 ms, repeat 5 times
                for _ in range(5):
                    winsound.Beep(750, 500)
                    time.sleep(0.1)
                return
            except (ImportError, AttributeError):
                # Not on Windows or winsound not available
                pass
                
            # Second option: try to play a sound file if possible
            try:
                # Try to use playsound if available
                from playsound import playsound
                try:
                    playsound("utils/sounds/alarm.wav", block=False)
                    time.sleep(3)  # Wait a bit for sound to play
                    return
                except Exception as sound_error:
                    logging.error(f"Error playing sound file: {sound_error}")
            except ImportError:
                # playsound not available
                logging.warning("playsound not installed, cannot play sound file")
                pass
                
            # Fallback to ASCII bell character
            for _ in range(10):
                print('\a')  # ASCII bell character
                time.sleep(0.3)
                
        except Exception as e:
            logging.error(f"Error playing emergency sound: {e}")
            # Ultimate fallback - just print the bell character
            for _ in range(5):
                print('\a')
    
    # No email functionality
