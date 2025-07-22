import os
import cv2
import face_recognition
import numpy as np
import logging

class FaceRecognizer:
    def __init__(self, known_faces_dir):
        """
        Initialize the face recognition module
        
        Args:
            known_faces_dir (str): Directory containing known face images
        """
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """
        Load all known faces from the known_faces directory
        """
        try:
            # Get all files in the directory
            for file_name in os.listdir(self.known_faces_dir):
                # Get only image files
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    # Get the person's name from the file name (remove extension)
                    name = os.path.splitext(file_name)[0]
                    
                    # Load the image
                    image_path = os.path.join(self.known_faces_dir, file_name)
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encoding
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        logging.info(f"Loaded face: {name}")
                    else:
                        logging.warning(f"No face found in {file_name}")
                        
            logging.info(f"Loaded {len(self.known_face_names)} known faces")
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")
    
    def recognize_faces(self, frame):
        """
        Recognize faces in a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of tuples (name, box) for each detected face
        """
        # Convert frame from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        
        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known face
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            face_names.append((name, (left, top, right, bottom)))
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        return face_names
