import tensorflow as tf
from tensorflow.keras import models

def model():
    try:
        model_cnn = tf.keras.models.load_model(r'C:\Users\Asus\Documents\Especializacion\proyecto1\UAO-Neumonia/WilhemNet_86.h5')#La ruta debe direccionar en donde esta el modelo de forma local.
        return model_cnn
    except OSError as e:
        print(f"Error loading model: {e}")
        return None
