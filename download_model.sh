#!/bin/bash

# Create weights directory
mkdir -p weights

echo "Choose a model source:"
echo "1. HongguLiu (recommended)"
echo "2. FaceForensics++ official"
echo "3. I have my own weights"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Visit: https://github.com/HongguLiu/Deepfake-Detection"
        echo "Download their pretrained weights and save to weights/xception_ff.pth"
        ;;
    2)
        echo "Visit: https://github.com/ondyari/FaceForensics"
        echo "Follow instructions to download and save to weights/xception_ff.pth"
        ;;
    3)
        read -p "Enter path to your weights file: " model_path
        cp "$model_path" weights/xception_ff.pth
        echo "✓ Copied to weights/xception_ff.pth"
        ;;
esac

echo ""
echo "After downloading, run:"
echo "  python test_detector.py --image test.jpg"