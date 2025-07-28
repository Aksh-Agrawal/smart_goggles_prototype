# Setting Up Smart Goggles on Raspberry Pi 4 Model B

This guide walks you through the process of setting up the Smart Goggles prototype on a Raspberry Pi 4 Model B.

## Hardware Requirements

- Raspberry Pi 4 Model B (recommended: 4GB or 8GB RAM version)
- MicroSD card (32GB or larger recommended)
- Power supply (3A USB-C recommended)
- Camera module (Raspberry Pi Camera Module or USB webcam)
- Optional: Speakers/headphones for audio output
- Optional: Microphone for voice input
- Optional: Portable power bank for mobile use
- Optional: Small display for debugging

## Initial Raspberry Pi Setup

1. **Install Raspberry Pi OS**:
   - Download Raspberry Pi Imager from [raspberrypi.com](https://www.raspberrypi.com/software/)
   - Insert your microSD card into your computer
   - Open Raspberry Pi Imager and select "Raspberry Pi OS (64-bit)" as the operating system
   - Select your microSD card as the storage
   - Click on the gear icon (⚙️) to access advanced options:
     - Enable SSH
     - Set username and password
     - Configure Wi-Fi credentials
   - Click "Write" to flash the OS to the microSD card

2. **Boot Up Your Raspberry Pi**:
   - Insert the microSD card into your Raspberry Pi
   - Connect peripherals (monitor, keyboard, mouse) if needed
   - Power on the Raspberry Pi
   - Complete the initial setup process

## Transferring the Smart Goggles Code

Choose one of these methods to transfer the code:

### Method 1: Using Git (Recommended)

1. **Install Git on Raspberry Pi**:
   ```bash
   sudo apt update
   sudo apt install -y git
   ```

2. **Clone the repository**:
   ```bash
   cd ~
   git clone https://github.com/Aksh-Agrawal/smart_goggles_prototype.git
   cd smart_goggles_prototype
   ```

### Method 2: Direct Transfer

1. **Using SCP (from your computer)**:
   ```bash
   # On Windows (PowerShell):
   scp -r D:\Codes\smart_goggles_prototype pi@raspberrypi.local:~/
   
   # On Linux/macOS:
   scp -r /path/to/smart_goggles_prototype pi@raspberrypi.local:~/
   ```

2. **Using a USB Drive**:
   - Copy the project folder to a USB drive
   - Connect the USB drive to Raspberry Pi
   - Mount and copy the files:
   ```bash
   sudo mkdir -p /mnt/usb
   sudo mount /dev/sda1 /mnt/usb  # Device name may vary
   cp -r /mnt/usb/smart_goggles_prototype ~/
   sudo umount /mnt/usb
   ```

## Installing Dependencies

1. **Install system dependencies**:
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv libopencv-dev libatlas-base-dev libhdf5-dev libhdf5-serial-dev libjasper-dev
   sudo apt install -y python3-opencv libqt5gui5 libqtgui4 libqt4-test
   ```

2. **Set up a Python virtual environment** (recommended):
   ```bash
   cd ~/smart_goggles_prototype
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Note: Some packages might need to be installed specifically for Raspberry Pi. If you encounter errors, try:
   ```bash
   pip install opencv-contrib-python-headless
   pip install https://github.com/Qengineering/tensorflow-2.4.0/releases/download/RPi-4/tensorflow-2.4.0-cp39-cp39-linux_aarch64.whl
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   nano .env  # Edit with your configuration
   ```

## Configuring the Camera

1. **For Raspberry Pi Camera Module**:
   ```bash
   sudo raspi-config
   ```
   Navigate to "Interface Options" > "Camera" and enable it.

2. **For USB webcam**:
   ```bash
   # Test if camera is detected
   ls -l /dev/video*
   ```

## Starting the Smart Goggles Application

1. **Run the application**:
   ```bash
   cd ~/smart_goggles_prototype
   source venv/bin/activate  # If using virtual environment
   python main.py
   ```

2. **To run on startup** (optional):
   ```bash
   crontab -e
   ```
   Add this line:
   ```
   @reboot cd ~/smart_goggles_prototype && source venv/bin/activate && python main.py
   ```

## Optimizing Performance

Raspberry Pi has limited resources compared to a desktop computer. Consider these optimizations:

1. **Reduce resolution**:
   Edit `main.py` to use a lower camera resolution:
   ```python
   # Example: Change from 640x480 to 320x240
   ```

2. **Reduce processing frequency**:
   Adjust frame processing rate in the code to reduce CPU usage.

3. **Overclocking** (for advanced users):
   ```bash
   sudo nano /boot/config.txt
   ```
   Add/modify these lines:
   ```
   over_voltage=6
   arm_freq=2000
   ```
   Warning: This may require additional cooling.

## Troubleshooting

### Camera Issues
- Make sure the camera ribbon cable is correctly connected
- Check camera permissions: `sudo usermod -a -G video $USER`
- Reboot after camera configuration changes

### Performance Issues
- Monitor system resources: `htop`
- Consider reducing model complexity or using quantized models
- Close other applications when running Smart Goggles

### Environment Variables
- Follow the instructions in `EMERGENCY_ALERTS_HELP.md` for setting up email/SMS alerts
- Make sure all required environment variables are set in the `.env` file

## Power Management

For portable use:
- Use a high-capacity power bank (10,000+ mAh)
- Consider using the `sudo poweroff` command before disconnecting power
- Monitor battery levels if using a battery monitoring HAT

## Additional Resources

- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [OpenCV on Raspberry Pi Guides](https://qengineering.eu/deep-learning-with-raspberry-pi-4.html)
- [TensorFlow Lite for Raspberry Pi](https://www.tensorflow.org/lite/guide/python)

---

For further assistance or to report issues, please contact the project maintainer.
