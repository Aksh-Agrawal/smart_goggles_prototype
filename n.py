import cv2
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

def setup_camera(use_smart_connect=False, width=640, height=480, fps=30):
    """
    Set up camera capture based on selected device (Smart Connect or Default)
    
    Args:
        use_smart_connect (bool): True to use Smart Connect camera (external), False to use internal webcam
        width (int): Frame width
        height (int): Frame height
        fps (int): Frames per second
        
    Returns:
        cv2.VideoCapture: Camera capture object
    """
    # Set camera ID based on choice
    if use_smart_connect:
        camera_id = 1  # Change to 2 if Smart Connect is at index 2
        logging.info("Trying to connect to Smart Connect camera...")
    else:
        camera_id = 0
        logging.info("Trying to connect to Default (laptop) webcam...")

    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        logging.error(f"❌ Failed to open camera with ID {camera_id}")
    else:
        logging.info(f"✅ Camera initialized with ID {camera_id}, resolution {width}x{height}, {fps} FPS")
    
    return cap


# ======= Example usage =======
if __name__ == "__main__":
    # Set to True to use Smart Connect camera
    use_smart_connect_camera = True

    cap = setup_camera(use_smart_connect=use_smart_connect_camera)

    # Run preview
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("⚠️ Failed to read frame from camera.")
            break

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) == 27:  # Press ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
