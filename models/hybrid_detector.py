import os
import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from .xception import XceptionNet

# Import your existing frequency analysis
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from image_detection import image_features, image_fake_score


class HybridDetector:
    """
    Combines pretrained CNN with your frequency analysis
    Backward compatible with your existing code
    """
    def __init__(self, model_path=None, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = None
        self.model_loaded = False
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Try to load model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            self.model = XceptionNet(num_classes=2).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('state_dict', 
                            checkpoint.get('model_state_dict', checkpoint))
            else:
                state_dict = checkpoint
            
            # Remove 'model.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k.replace('model.', '')] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            self.model_loaded = True
            print(f"✓ Loaded CNN model from {model_path}")
            print(f"  Device: {self.device}")
            return True
        except Exception as e:
            print(f"⚠ Could not load CNN model: {e}")
            self.model = None
            self.model_loaded = False
            return False
    
    def predict(self, image_bgr, mode='hybrid', weights=None):
        """
        Unified prediction interface
        
        Args:
            image_bgr: OpenCV BGR image
            mode: 'hybrid', 'cnn', or 'frequency'
            weights: {'cnn': 0.65, 'freq': 0.35} for hybrid mode
        
        Returns:
            dict with scores and metadata
        """
        if weights is None:
            weights = {'cnn': 0.65, 'freq': 0.35}
        
        result = {
            'cnn_score': None,
            'frequency_score': None,
            'ensemble_score': None,
            'prediction': 'unknown',
            'confidence': 'low',
            'features': None,
            'explanation': []
        }
        
        # Frequency analysis (always available)
        if mode in ['hybrid', 'frequency']:
            try:
                result['features'] = image_features(image_bgr)
                result['frequency_score'] = image_fake_score(result['features'])
            except Exception as e:
                print(f"Frequency analysis error: {e}")
        
        # CNN analysis (if model available)
        if mode in ['hybrid', 'cnn'] and self.model_loaded:
            try:
                result['cnn_score'] = self._predict_cnn(image_bgr)
            except Exception as e:
                print(f"CNN prediction error: {e}")
        
        # Compute final score
        if mode == 'hybrid':
            if result['cnn_score'] is not None and result['frequency_score'] is not None:
                result['ensemble_score'] = (
                    weights['cnn'] * result['cnn_score'] + 
                    weights['freq'] * result['frequency_score']
                )
            elif result['cnn_score'] is not None:
                result['ensemble_score'] = result['cnn_score']
                result['explanation'].append("CNN only (frequency unavailable)")
            else:
                result['ensemble_score'] = result['frequency_score']
                result['explanation'].append("Frequency only (no CNN model)")
        elif mode == 'cnn':
            result['ensemble_score'] = result['cnn_score']
        else:  # frequency
            result['ensemble_score'] = result['frequency_score']
        
        # Final prediction
        if result['ensemble_score'] is not None:
            result['prediction'] = 'FAKE' if result['ensemble_score'] > 0.5 else 'REAL'
            
            # Confidence assessment
            score = result['ensemble_score']
            if mode == 'hybrid' and result['cnn_score'] and result['frequency_score']:
                agreement = abs(result['cnn_score'] - result['frequency_score'])
                if (score > 0.7 or score < 0.3) and agreement < 0.2:
                    result['confidence'] = 'high'
                elif score > 0.6 or score < 0.4:
                    result['confidence'] = 'medium'
            else:
                if score > 0.7 or score < 0.3:
                    result['confidence'] = 'medium'
        
        # Add explanations from features
        if result['features']:
            self._add_explanations(result)
        
        return result
    
    def _predict_cnn(self, image_bgr):
        """Internal CNN prediction"""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            return float(probs[0][1].item())
    
    def _add_explanations(self, result):
        """Add human-readable explanations"""
        f = result['features']
        if f['block'] > 0.25:
            result['explanation'].append("High compression detected")
        if f['gfx'] > 0.5:
            result['explanation'].append("Graphics overlay detected")
        if f['peak'] > 2.5:
            result['explanation'].append("Unusual frequency peaks (GAN artifact)")
        if f['grid'] > 0.15:
            result['explanation'].append("Grid pattern (upsampling artifact)")
        if f['benf'] > 0.2:
            result['explanation'].append("Benford's law violation")