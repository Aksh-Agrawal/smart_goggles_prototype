import googlemaps
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

class NavigationHelper:
    def __init__(self, api_key=None):
        """
        Initialize the navigation helper with Google Maps API
        
        Args:
            api_key (str): Google Maps API key
        """
        # Try to load API key from environment variable if not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("GOOGLE_MAPS_API_KEY")
            
        if api_key:
            try:
                self.gmaps = googlemaps.Client(key=api_key)
                self.api_available = True
            except Exception as e:
                logging.error(f"Error initializing Google Maps client: {e}")
                self.api_available = False
        else:
            logging.warning("No Google Maps API key provided. Navigation features will be disabled.")
            self.api_available = False
    
    def get_directions(self, origin, destination, mode="walking"):
        """
        Get directions from origin to destination
        
        Args:
            origin (str): Starting location
            destination (str): Destination location
            mode (str): Transportation mode (walking, driving, transit, bicycling)
            
        Returns:
            list: List of step-by-step directions
        """
        if not self.api_available:
            return ["Navigation service is not available. Please configure Google Maps API key."]
            
        try:
            # Request directions
            now = datetime.now()
            directions_result = self.gmaps.directions(origin, destination, mode=mode, departure_time=now)
            
            if not directions_result:
                return ["No directions found between these locations."]
                
            # Extract step-by-step instructions
            steps = []
            for leg in directions_result[0]['legs']:
                for step in leg['steps']:
                    # Extract the instruction, removing HTML tags
                    instruction = step['html_instructions']
                    instruction = instruction.replace('<b>', '').replace('</b>', '')
                    instruction = instruction.replace('<div style="font-size:0.9em">', '. ').replace('</div>', '')
                    steps.append(instruction)
                    
            return steps
        except Exception as e:
            logging.error(f"Error fetching directions: {e}")
            return ["Error fetching directions. Please try again later."]
