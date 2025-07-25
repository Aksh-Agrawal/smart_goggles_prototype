"""
Smart Connect camera integration for Smart Goggles

This module provides functionality to connect to and stream video from the
Smart Connect app, allowing the Smart Goggles to use a smartphone camera.
"""

import cv2
import numpy as np
import requests
import socket
import logging
import time
import threading
from urllib.parse import urlparse

class SmartConnectCamera:
    def __init__(self, ip_address=None, port=8080, path="/video", timeout=10):
        """
        Initialize the Smart Connect camera interface
        
        Args:
            ip_address (str, optional): IP address of the Smart Connect device
                                        If None, will attempt auto-discovery
            port (int): Port number of the video stream
            path (str): URL path to the video stream
            timeout (int): Connection timeout in seconds
        """
        self.ip_address = ip_address
        self.port = port
        self.path = path
        self.timeout = timeout
        self.stream_url = None
        self.video_capture = None
        self.connected = False
        self.discovery_thread = None
        self.auto_reconnect = True
        
        # Cached frame (used when connection is lost)
        self.last_frame = None
        self.frame_timestamp = 0
        
        logging.info("Smart Connect camera module initialized")
    
    def discover_device(self):
        """
        Attempt to discover the Smart Connect device on the local network
        
        Returns:
            bool: True if device was discovered, False otherwise
        """
        if self.ip_address:
            return True
            
        logging.info("Attempting to discover Smart Connect device...")
        
        # Create UDP socket for broadcast
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(2.0)
        
        # Discovery message
        discovery_message = "SMART_GOGGLES_DISCOVERY"
        
        # Try common subnet broadcast addresses
        broadcast_addresses = ["192.168.1.255", "192.168.0.255", "10.0.0.255", "255.255.255.255"]
        
        for addr in broadcast_addresses:
            try:
                sock.sendto(discovery_message.encode(), (addr, 8888))
            except:
                pass
        
        # Listen for responses
        start_time = time.time()
        while time.time() - start_time < 5.0:  # 5 second timeout
            try:
                data, addr = sock.recvfrom(1024)
                if data.decode().startswith("SMART_CONNECT"):
                    self.ip_address = addr[0]
                    logging.info(f"Smart Connect device discovered at {self.ip_address}")
                    return True
            except socket.timeout:
                pass
            except Exception as e:
                logging.error(f"Error during discovery: {e}")
        
        logging.warning("No Smart Connect device found on the network")
        return False
    
    def start_discovery_in_background(self):
        """Start device discovery in a background thread"""
        if self.discovery_thread and self.discovery_thread.is_alive():
            return
            
        self.discovery_thread = threading.Thread(target=self._background_discovery)
        self.discovery_thread.daemon = True
        self.discovery_thread.start()
        
    def _background_discovery(self):
        """Background thread to continuously attempt device discovery"""
        while not self.connected and self.auto_reconnect:
            if self.discover_device():
                self.connect()
                break
            time.sleep(5)  # Wait before retrying
    
    def connect(self):
        """
        Connect to the Smart Connect camera stream
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if not self.ip_address:
            success = self.discover_device()
            if not success:
                return False
        
        # Construct the stream URL
        self.stream_url = f"http://{self.ip_address}:{self.port}{self.path}"
        
        try:
            # Test the connection with a HEAD request
            response = requests.head(self.stream_url, timeout=2)
            if response.status_code >= 400:
                logging.error(f"Failed to connect to Smart Connect: HTTP {response.status_code}")
                return False
                
            # Initialize OpenCV VideoCapture with the stream URL
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            if not self.video_capture.isOpened():
                logging.error("Failed to open Smart Connect video stream")
                return False
                
            # Test read a frame
            ret, frame = self.video_capture.read()
            if not ret:
                logging.error("Failed to read frame from Smart Connect")
                self.video_capture.release()
                self.video_capture = None
                return False
                
            self.connected = True
            self.last_frame = frame
            self.frame_timestamp = time.time()
            logging.info(f"Successfully connected to Smart Connect camera at {self.stream_url}")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting to Smart Connect: {e}")
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            return False
    
    def disconnect(self):
        """Disconnect from the Smart Connect camera"""
        self.auto_reconnect = False
        self.connected = False
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        logging.info("Disconnected from Smart Connect camera")
    
    def read(self):
        """
        Read a frame from the Smart Connect camera
        
        Returns:
            tuple: (success (bool), frame (numpy.ndarray))
        """
        if not self.connected or not self.video_capture:
            # If we have a cached frame less than 5 seconds old, return it
            if self.last_frame is not None and time.time() - self.frame_timestamp < 5:
                return True, self.last_frame.copy()
            
            # Otherwise, try to reconnect
            if self.auto_reconnect and not self.discovery_thread:
                self.start_discovery_in_background()
            
            # Return a black frame as fallback
            return False, np.zeros((480, 640, 3), dtype=np.uint8)
        
        try:
            ret, frame = self.video_capture.read()
            
            if ret:
                self.last_frame = frame
                self.frame_timestamp = time.time()
                return True, frame
            else:
                logging.warning("Failed to read frame from Smart Connect")
                self.connected = False
                
                # Try to reconnect
                if self.auto_reconnect:
                    threading.Thread(target=self.connect).start()
                
                # Return the last good frame if we have one
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
                else:
                    return False, np.zeros((480, 640, 3), dtype=np.uint8)
                    
        except Exception as e:
            logging.error(f"Error reading from Smart Connect: {e}")
            self.connected = False
            
            # Try to reconnect
            if self.auto_reconnect:
                threading.Thread(target=self.connect).start()
            
            # Return the last good frame if we have one
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            else:
                return False, np.zeros((480, 640, 3), dtype=np.uint8)
    
    def is_connected(self):
        """
        Check if the camera is connected
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected
    
    def set_ip_address(self, ip_address):
        """
        Set the IP address of the Smart Connect device
        
        Args:
            ip_address (str): IP address to connect to
        """
        self.ip_address = ip_address
        if self.connected:
            self.disconnect()
            
    def set_auto_reconnect(self, auto_reconnect):
        """
        Set whether to automatically reconnect if the connection is lost
        
        Args:
            auto_reconnect (bool): Whether to auto reconnect
        """
        self.auto_reconnect = auto_reconnect
