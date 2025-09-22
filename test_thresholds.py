# Threshold testing with updated conservative approach for Tamim detection

import tensorflow as tf
import numpy as np
import os


model = tf.keras.models.load_model('final_tamim_classifier.keras')

def predict_new_image(image_path, model, threshold=0.5):
    """Predict on a new image"""
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, 0)
        
        prediction_prob = model.predict(image, verbose=0)[0][0]
        prediction_class = "Tamim" if prediction_prob > threshold else "Not_Tamim"
        confidence = prediction_prob if prediction_prob > threshold else 1 - prediction_prob
        
        return {
            'prediction': prediction_class,
            'confidence': confidence,
            'raw_probability': prediction_prob
        }
    except Exception as e:
        return {'error': str(e)}

test_images = {
    'test.jpeg': 'test.jpeg',
    'test2.jpeg': 'test2.jpeg',
    'test3.png': 'test3.png'
}

print("THRESHOLD OPTIMIZATION FOR TAMIM DETECTION")

thresholds = [0.45, 0.5, 0.55, 0.6, 0.65]

for threshold in thresholds:
    correct = 0
    print(f"\nThreshold: {threshold}")
    
    for img_name, img_path in test_images.items():
        if os.path.exists(img_path):
            result = predict_new_image(img_path, model, threshold=threshold)
            if 'error' not in result:
                is_correct = result['prediction'] == 'Tamim'  
                status = "✅" if is_correct else "❌"
                print(f"   {img_name}: {result['prediction']} {status} (prob: {result['raw_probability']:.3f})")
                if is_correct:
                    correct += 1
    
    accuracy = correct / len(test_images) * 100
    print(f"   Accuracy: {correct}/{len(test_images)} = {accuracy:.0f}%")
