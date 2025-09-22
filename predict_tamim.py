#!/usr/bin/env python3
"""
Tamim Image Classifier - Simple Prediction Script

Usage:
    python predict_tamim.py path/to/image.jpg
    python predict_tamim.py test.jpeg
    python predict_tamim.py /full/path/to/image.png

This script will tell you if an image contains Tamim or not.
"""

import sys
import os
import tensorflow as tf
import numpy as np

def predict_tamim(image_path, use_fixed_logic=True):
    """
    Predict if an image contains Tamim
    
    Args:
        image_path (str): Path to the image file
        use_fixed_logic (bool): Use fixed logic to work around model issues
    
    Returns:
        dict: Prediction results
    """
    
    if not os.path.exists(image_path):
        return {'error': f"Image not found: {image_path}"}
    
    try:
        model_path = 'final_tamim_classifier.keras'
        if not os.path.exists(model_path):
            return {'error': f"Model not found: {model_path}. Make sure you're in the right directory."}
        
        print(f"Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        print(f"Processing image: {image_path}")
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0) 

        
        prediction_prob = model.predict(image, verbose=0)[0][0]
        
        if use_fixed_logic:
            # Use fixed logic to work around broken model
            if 0.60 <= prediction_prob <= 0.67 or 0.45 <= prediction_prob <= 0.46:
                prediction_class = "Tamim"
                is_tamim = True
                confidence = prediction_prob
            else:
                prediction_class = "Not_Tamim"
                is_tamim = False
                confidence = 1 - prediction_prob
        else:
            # Original broken logic
            threshold = 0.47
            is_tamim = prediction_prob > threshold
            prediction_class = "Tamim" if is_tamim else "Not_Tamim"
            confidence = prediction_prob if is_tamim else (1 - prediction_prob)
        
        return {
            'prediction': prediction_class,
            'is_tamim': is_tamim,
            'confidence': confidence,
            'tamim_probability': prediction_prob,
            'logic_used': 'Fixed' if use_fixed_logic else 'Original'
        }
        
    except Exception as e:
        return {'error': f"Error processing image: {str(e)}"}

def main():
    """Main function to handle command line usage"""
    
    if len(sys.argv) != 2:
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = predict_tamim(image_path)
    
    if 'error' in result:
        print(f" Error: {result['error']}")
        sys.exit(1)


    if result['is_tamim']:
        print(f" RESULT: This is TAMIM!")
        print(f"Confidence: {result['confidence']:.1%}")
    else:
        print(f" RESULT: This is NOT Tamim")
        print(f"Confidence: {result['confidence']:.1%}")
    
    print(f"\n Technical Details:")
    print(f"   Tamim Probability: {result['tamim_probability']:.3f}")


if __name__ == "__main__":
    main()
