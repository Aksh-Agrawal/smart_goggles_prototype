#!/bin/bash
# Setup script for Smart Goggles prototype on Linux/macOS

echo -e "\e[32mSetting up Smart Goggles Prototype...\e[0m"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\e[33mCreating virtual environment...\e[0m"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "\e[33mActivating virtual environment...\e[0m"
source venv/bin/activate

# Install required packages
echo -e "\e[33mInstalling required packages (this may take a while)...\e[0m"
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "\e[33mCreating .env file from example...\e[0m"
    cp .env.example .env
    echo -e "\e[36mPlease edit the .env file to set up your API keys and configuration\e[0m"
fi

# Create known_faces directory if it doesn't exist
if [ ! -d "known_faces" ]; then
    echo -e "\e[33mCreating known_faces directory...\e[0m"
    mkdir -p known_faces
    echo -e "\e[36mAdd face images to the known_faces directory (name each file with the person's name)\e[0m"
fi

echo -e "\e[32mSetup complete!\e[0m"
echo -e "\e[32mTo run the Smart Goggles prototype: python main.py\e[0m"
