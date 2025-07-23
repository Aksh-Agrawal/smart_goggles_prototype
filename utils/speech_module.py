import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os
import playsound
import tempfile
import threading
import time
import queue
import logging

class SpeechModule:
    def __init__(self, use_gtts=False, language="en-IN"):
        """
        Initialize speech recognition and text-to-speech modules
        
        Args:
            use_gtts (bool): Whether to use gTTS (True) or pyttsx3 (False) for TTS
            language (str): Language code for speech (default: Indian English)
        """
        self.language = language
        self.recognizer = sr.Recognizer()
        self.use_gtts = use_gtts
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = threading.Thread(target=self._speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()
        
        # Initialize pyttsx3 if not using gTTS
        if not use_gtts:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 180)  # Adjust speaking rate
                voices = self.engine.getProperty('voices')
                
                # First try to find an Indian English voice
                indian_voice_found = False
                for voice in voices:
                    if "india" in voice.name.lower() or "hindi" in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        indian_voice_found = True
                        logging.info(f"Using Indian voice: {voice.name}")
                        break
                
                # If no Indian voice, fallback to female voice
                if not indian_voice_found:
                    for voice in voices:
                        if "female" in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            logging.info(f"Using female voice: {voice.name}")
                            break
            except Exception as e:
                logging.error(f"Error initializing pyttsx3: {e}")
                self.use_gtts = True  # Fallback to gTTS
    
    def _speech_worker(self):
        """Background thread to process speech queue to prevent blocking main thread"""
        while True:
            if not self.speech_queue.empty():
                text = self.speech_queue.get()
                self.is_speaking = True
                self._speak_now(text)
                self.is_speaking = False
                self.speech_queue.task_done()
            else:
                time.sleep(0.1)
    
    def speak(self, text, priority=False):
        """
        Convert text to speech
        
        Args:
            text (str): Text to speak
            priority (bool): If True, clear queue and speak immediately
        """
        if not text:
            return
            
        if priority:
            # Clear the queue for priority messages
            while not self.speech_queue.empty():
                self.speech_queue.get()
                self.speech_queue.task_done()
                
        self.speech_queue.put(text)
    
    def _speak_now(self, text):
        """
        Actual speech synthesis (called by worker thread)
        
        Args:
            text (str): Text to speak
        """
        try:
            if self.use_gtts:
                # Use Google Text-to-Speech with preferred language (default: Indian English)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    temp_filename = tmp_file.name
                
                # Parse the language code to get the base language
                lang_base = self.language.split('-')[0] if '-' in self.language else self.language
                # Use Indian TLD for Indian English
                tld = 'co.in' if self.language.lower() == 'en-in' else 'com'
                
                tts = gTTS(text=text, lang=lang_base, tld=tld)
                tts.save(temp_filename)
                playsound.playsound(temp_filename)
                os.remove(temp_filename)
            else:
                # Use pyttsx3
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            logging.error(f"Error in speech synthesis: {e}")
            
    def listen(self, timeout=5, phrase_time_limit=5, language=None):
        """
        Listen for speech and convert to text
        
        Args:
            timeout (int): How long to wait for speech to start (seconds)
            phrase_time_limit (int): Maximum duration for speech input (seconds)
            language (str): Language code for speech recognition (override instance language)
            
        Returns:
            str: Recognized text or empty string if not recognized
        """
        with sr.Microphone() as source:
            try:
                # Enhanced ambient noise adjustment (longer duration)
                logging.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                # Adjust energy threshold for better speech detection
                # Increase this value in noisy environments
                self.recognizer.energy_threshold = 4000  # Default is 300
                self.recognizer.dynamic_energy_threshold = True
                
                self.speak("Listening...", priority=True)
                
                try:
                    # Listen with a slightly longer pause threshold
                    self.recognizer.pause_threshold = 0.8  # Default is 0.8, increase for slower speech
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                    
                    # Try multiple recognition services in order
                    # Use provided language or fall back to instance language
                    recognition_language = language if language else self.language
                    text = self._try_multiple_recognition_services(audio, recognition_language)
                    
                    if text:
                        return text.lower()
                    else:
                        # Give feedback only if audio was captured but not recognized
                        logging.warning("Speech was not recognized")
                        return ""
                        
                except sr.UnknownValueError:
                    logging.warning("Speech was not recognized")
                    return ""
            except sr.WaitTimeoutError:
                logging.info("No speech detected within timeout period")
                return ""
            except Exception as e:
                logging.error(f"Error in speech recognition: {e}")
                return ""
                
    def _try_multiple_recognition_services(self, audio, language="en-IN"):
        """
        Try multiple speech recognition services in order of reliability
        
        Args:
            audio: Audio data to recognize
            language (str): Language code for recognition
            
        Returns:
            str: Recognized text or empty string
        """
        # List of recognition methods to try in order
        recognition_methods = [
            # Method 1: Google Web Speech API (most reliable but needs internet)
            lambda: self.recognizer.recognize_google(audio, language=language),
            
            # Method 2: Try with a different language variant if original is en-IN
            lambda: self.recognizer.recognize_google(audio, language="en-GB") if language == "en-IN" else None,
            
            # Method 3: Try Sphinx (offline, less accurate but doesn't need internet)
            lambda: self.recognizer.recognize_sphinx(audio) if hasattr(self.recognizer, 'recognize_sphinx') else None
        ]
        
        # Try each method in order
        for method in recognition_methods:
            try:
                result = method()
                if result:
                    return result
            except (sr.UnknownValueError, sr.RequestError):
                continue
            except Exception as e:
                logging.error(f"Error in recognition method: {e}")
                continue
                
        # If all methods fail, return empty string
        return ""
    
    def play_alert_sound(self, sound_type="warning"):
        """
        Play a predefined alert sound
        
        Args:
            sound_type (str): Type of alert sound to play ("warning", "emergency", etc.)
        """
        # Generate a simple beep sound using ASCII bell
        if sound_type == "emergency":
            # For emergency, play a louder, longer sound
            print('\a\a\a')  # Multiple bell characters
        else:
            # For warning, play a single beep
            print('\a')
        
        # In a real implementation, you'd use playsound to play actual audio files
