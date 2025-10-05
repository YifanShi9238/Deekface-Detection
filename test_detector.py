#!/usr/bin/env python3
"""
Quick test script for hybrid detector
"""
import cv2
import argparse
from models.hybrid_detector import HybridDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Test image path')
    parser.add_argument('--model', default='weights/xception_ff.pth')
    args = parser.parse_args()
    
    # Load
    detector = HybridDetector(model_path=args.model)
    image = cv2.imread(args.image)
    
    if image is None:
        print(f"Error: Cannot load {args.image}")
        return
    
    # Test all modes
    for mode in ['hybrid', 'frequency', 'cnn']:
        print(f"\n{'='*60}")
        print(f"Mode: {mode.upper()}")
        print('='*60)
        
        result = detector.predict(image, mode=mode)
        
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Score: {result['ensemble_score']:.3f}" if result['ensemble_score'] is not None else "Score: N/A")
        
        if result['explanation']:
            print("\nDetails:")
            for exp in result['explanation']:
                print(f"  • {exp}")

if __name__ == '__main__':
    main()