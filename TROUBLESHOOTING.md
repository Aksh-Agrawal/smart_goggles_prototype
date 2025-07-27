# Troubleshooting Guide for Smart Goggles

## Scene Description Not Working

If the scene description feature isn't working with the Gemini API:

1. **Get a Valid Gemini API Key**:

   - Visit https://aistudio.google.com/app/apikey
   - Create a free account if you don't have one
   - Generate a new API key
   - Copy your API key

2. **Update Your .env File**:

   - Open the `.env` file in the project root
   - Replace the `OPENAI_API_KEY` value with your newly generated key:
     ```
     OPENAI_API_KEY=your_actual_api_key_here
     ```
   - Set the vision model type in .env:
     ```
     VISION_MODEL=gemini  # or 'openai' if using OpenAI
     ```

3. **Check Logs for Errors**:

   - When you press 'S' to describe the scene, check the console output
   - Look for error messages related to the Gemini API
   - Common issues include:
     - Invalid API key
     - Image size too large (now handled automatically)
     - API rate limits exceeded (free tier has limitations)
     - Model version deprecation (the application now uses gemini-1.5-flash instead of the deprecated gemini-pro-vision)

4. **Test Gemini API Separately**:
   - Consider testing your API key with a simple script before using it in the app
   - This helps confirm your key is working correctly

## Emergency Alerts Not Being Sent

If emergency alerts (email/SMS) aren't working:

1. **Email Configuration**:

   a. **Gmail Setup**:

   - You need to use an "App Password" if using Gmail
   - Go to your Google Account → Security → App passwords
   - Generate an app password specifically for your application
   - Update the `.env` file with your email and app password:
     ```
     EMAIL_SENDER=your_email@gmail.com
     EMAIL_PASSWORD=your_app_password
     EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
     ```

   b. **SMTP Settings**:

   - Verify your SMTP server and port are correct
   - For Gmail:
     ```
     EMAIL_SERVER=smtp.gmail.com
     EMAIL_PORT=587
     ```

2. **SMS Configuration**:

   a. **Twilio Setup**:

   - Sign up for a Twilio account at https://www.twilio.com
   - Get a Twilio phone number
   - Find your Account SID and Auth Token in the Twilio Dashboard
   - Update your `.env` file:
     ```
     TWILIO_ACCOUNT_SID=your_account_sid
     TWILIO_AUTH_TOKEN=your_auth_token
     TWILIO_FROM_NUMBER=your_twilio_number  # Format: +1234567890
     TWILIO_TO_NUMBERS=recipient_number1,recipient_number2  # Format: +1234567890
     ```

   b. **Install Twilio Package**:

   - Run: `pip install twilio`

3. **Test Your Settings**:
   - After updating your configuration, press 'E' to trigger an emergency alert
   - Check the console for detailed logs about the sending process
   - Look for specific error messages that may indicate what's wrong

## Object Detection Issues

1. **Objects Not Being Detected**:

   - Check lighting conditions - ensure there is sufficient light
   - Make sure objects are within the camera's field of view
   - Try adjusting the confidence threshold in `.env`:
     ```
     CONFIDENCE_THRESHOLD=0.4  # Lower the threshold to detect more objects
     ```
   - Press 'C' to toggle the center region visualization to see what counts as "in front of you"

2. **Voice Commands Not Detecting Objects**:

   - Speak clearly using phrases like "detect objects", "what do you see", or "identify objects"
   - Make sure the microphone is working and has proper permissions
   - Check that objects are in the center region (toggle with 'C' key to visualize)

3. **Wrong Object Labels**:

   - The default YOLOv8n model has limitations - consider using a larger model:
     ```
     YOLO_MODEL=yolov8m.pt  # Medium-sized model with better accuracy
     ```
   - Custom objects may not be recognized if not in the COCO dataset

## General Debugging Tips

1. **Check Log Messages**:

   - The app has extensive logging for easier troubleshooting
   - Look for ERROR and WARNING level messages in the console and smart_goggles.log file

2. **Verify Environment Variables**:

   - Make sure your `.env` file is in the project root directory
   - Ensure there are no typos in the variable names
   - See `.env.example` for the correct format and required variables

3. **Test Services Separately**:

   - Try sending a test email using a simple Python script
   - Test your API keys in their respective platforms (Google AI Studio, OpenAI, Azure, etc.)
   - Test camera using another application

4. **Network Connectivity**:
   - Ensure your computer has internet access for cloud-based features
   - Check firewall settings if API calls are failing
   - Check if any firewalls are blocking outgoing connections

If you continue experiencing issues after following these steps, please check the console output for specific error messages and refer to the relevant API documentation.
