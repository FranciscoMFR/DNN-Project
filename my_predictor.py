import tensorflow as tf
import numpy as np


def predict_with_model(model, img_path):

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [60,60])  #(60,60,3)
    image = tf.expand_dims(image, axis=0)    #(1,60,60,3)

    predictions = model.predict(image)       #[0.005 0.0003 0.99 0.00 ...]
    predictions = np.argmax(predictions)     #2 

    return predictions

if __name__ == "__main__":

    img_path = "C:\\Users\\User\\OneDrive - Universidade de Aveiro\\Desktop\\UA\\Projeto\\DNN-Project\\archive\\Test\\0\\01279.png"
    
    model = tf.keras.models.load_model("./Models")
    prediction = predict_with_model(model, img_path)

    print(f"prediction = {prediction}")