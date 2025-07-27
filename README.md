# Smart Goggles Prototype

A Python-based AI prototype for smart glasses to assist visually impaired users. This system uses computer vision (YOLOv8), keyboard commands, and speech feedback to provide real-time assistance.

## Features

### Core Features

- **Real-time object detection** using YOLOv8
- **Keyboard command interface** with on-screen instructions and text-to-speech
- **Face recognition** for identifying known individuals
- **Proximity alerts** for nearby objects
- **Enhanced OCR capability** to read text in the environment with multiple processing methods
- **Smart text detection** that automatically identifies text-containing objects

### Additional Features

- **Scene description** using AI vision models (Gemini Vision or GPT-4 Vision)
- **Emergency alert system** with email and SMS notifications

## Project Structure

```
smart_goggles_prototype/
│
├── main.py                   # Main application entry point
├── requirements.txt          # Dependencies
├── .env.example              # Example configuration file
│
├── known_faces/              # Directory for storing known face images
│
└── utils/                    # Utility modules
    ├── __init__.py           # Package initialization
    ├── object_detection.py   # YOLOv8 object detection with text detection capabilities
    ├── speech_module.py      # Speech recognition and synthesis
    ├── face_recognition_module.py # Face recognition module
    ├── ocr_module.py         # Enhanced Optical Character Recognition with multi-method processing
    ├── emergency_system.py   # Emergency alerts
    ├── scene_summarizer.py   # AI-powered scene description
    └── helpers.py            # Utility functions and configuration
```

## Setup

### Prerequisites

1. Python 3.8+ installed
2. Webcam or camera
3. Tesseract OCR installed (for text recognition)
4. The following API keys (optional, for enhanced functionality):
   - Gemini Vision API key or OpenAI API key for scene descriptions
   - Twilio account credentials for SMS alerts

### Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/smart_goggles_prototype.git
cd smart_goggles_prototype
```

2. Create and activate a virtual environment (recommended):

```
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Create a configuration file:

```
cp .env.example .env
```

5. Edit the `.env` file to set up your configuration and API keys.

6. Add face images to the `known_faces` directory:
   - Name each image file with the person's name (e.g., `john.jpg`)

## Usage

### Running the Prototype

```
python main.py
```

### Operation Guide

1. **Starting the System**:

   - Run the main.py script to start the Smart Goggles
   - The system will announce "Smart Goggles activated. Use keyboard commands to control the system."
   - A help window will appear showing all available keyboard commands
   - A window will open showing the camera feed with object detection overlays and keyboard command instructions

2. **Continuous Features**:

   - Object detection runs continuously in the background
   - Proximity alerts trigger automatically when objects get too close
   - Face recognition runs continuously to identify known people

3. **Using Keyboard Commands**:

   - Press the corresponding key to activate a feature (commands are shown on-screen)
   - The system will respond with voice feedback and visual cues
   - Each command is processed immediately when the key is pressed

4. **Stopping the System**:
   - Press 'Q' to exit the application

### Keyboard Commands

#### Available Keys:

- **O** - Detect and describe objects in the center of the field of view
- **F** - Identify faces in view (only announces unknown people; remains silent for known faces)
- **R** - Read text visible in the camera view using enhanced OCR with automatic text region detection
- **S** - Provide an AI-generated description of the scene
- **E** - Activate emergency mode with alerts
- **H** - Display help screen with command instructions
- **Q** - Quit the application

#### How Keyboard Commands Work:

1. All available commands are displayed in the upper-left corner of the screen
2. The keys are case-insensitive (both uppercase and lowercase work)
3. Commands provide voice feedback after they're activated
4. The help screen (H key) shows more detailed instructions
5. Keyboard commands are processed immediately without the need to wait for voice recognition

## Customization

### Adding Known Faces

1. Take a clear photo of a person's face
2. Save it in the `known_faces` directory with the person's name as the filename (e.g., `john.jpg`)

### Changing YOLOv8 Model

You can switch between different YOLOv8 models based on your performance needs:

- `yolov8n.pt`: Smallest model, fastest but less accurate
- `yolov8s.pt`: Small model, good balance of speed and accuracy
- `yolov8m.pt`: Medium model, more accurate but slower
- `yolov8l.pt`: Large model, very accurate but slower
- `yolov8x.pt`: Extra large model, most accurate but slowest

Update the `YOLO_MODEL` variable in your `.env` file to change the model.

### Customizing Keyboard Commands

To customize the keyboard commands or add new ones:

1. Open `main.py` and locate the `handle_keyboard_input` method
2. Add or modify the key bindings to recognize different keys
3. For example, to add a new command to announce the current time:

```python
# Add this to the handle_keyboard_input method
elif key == ord('t'):  # 'T' key for time
    current_time = datetime.now().strftime("%I:%M %p")
    self.speech.speak(f"The current time is {current_time}")
    # Add to command overlay
    self.add_command_to_overlay("T: Time")
```

You can also adjust voice settings (speech rate, voice type) in the `SpeechModule` class in `utils/speech_module.py`.

## OCR and Text Detection Features

### Enhanced OCR Capabilities

The Smart Goggles now include advanced OCR processing that:

1. **Automatically detects text-containing regions** in the environment
2. **Applies multiple preprocessing methods** to improve text recognition:
   - Standard OCR with preprocessing
   - Contrast-enhanced processing for difficult lighting
   - Adaptive thresholding for challenging text
3. **Integrates with object detection** to identify objects likely to contain text
4. **Provides visual feedback** highlighting detected text with confidence scores

### Using OCR Mode

1. Press the **R** key to activate OCR mode
2. Point the camera at text you want to read
3. Hold the camera steady for 1-2 seconds
4. The system will:
   - Highlight detected text regions
   - Read the text aloud via speech synthesis
   - Display the detected text on screen
5. OCR mode remains active for 5-8 seconds (longer if text is detected)

### OCR Configuration

You can customize OCR behavior by editing the `.env` file:

```
# OCR Service Selection (options: tesseract, azure, aws)
OCR_SERVICE=tesseract

# API Keys for cloud OCR (if using)
AZURE_VISION_KEY=your_key_here
AZURE_VISION_ENDPOINT=your_endpoint_here
AWS_ACCESS_KEY=your_key_here
AWS_SECRET_KEY=your_secret_here
```

## Troubleshooting

### Common Issues

1. **Keyboard Commands Not Working**:

   - Ensure the camera window is in focus (click on it)
   - Check that you're using the correct keys as shown on screen
   - Verify that Caps Lock is not causing issues
   - Restart the application if commands become unresponsive

2. **Camera Issues**:

   - Ensure your webcam is connected and functioning
   - Try changing the `CAMERA_ID` in your `.env` file if you have multiple cameras
   - Adjust `FRAME_WIDTH` and `FRAME_HEIGHT` for better performance

3. **Performance Issues**:
   - Try using a smaller YOLOv8 model (yolov8n.pt)
   - Reduce the camera resolution in your `.env` file
   - Close other resource-intensive applications
4. **Face Recognition Problems**:

   - Ensure good lighting conditions
   - Use clear, front-facing photos in the `known_faces` directory
   - Add multiple photos of the same person with different angles/lighting

5. **API Integration Issues**:
   - Double-check your API keys in the `.env` file
   - Ensure you have an active internet connection for cloud-based features

## Future Improvements

1. **Hardware Integration**: Adapt for Raspberry Pi with Pi Camera and speakers
2. **Enhanced Navigation**: Incorporate depth sensing for better obstacle detection
3. **More Advanced AI**: Integrate multimodal models for richer understanding
4. **Customizable Keyboard Layout**: Allow users to configure their preferred key bindings
5. **Offline Support**: Implement offline models for core functionality
6. **Accessibility Options**: Add support for alternative input methods (e.g., joystick, special controllers)
7. **Voice Command Option**: Re-introduce improved voice commands as an alternative to keyboard input

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3)
- [pytesseract](https://github.com/madmaze/pytesseract)
