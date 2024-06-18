import os
import numpy as np
from PIL import Image
import keras
from keras.models import load_model
from keras.preprocessing import image
import shutil

model = load_model('modelzz.h5')

def classify_and_move(images, sorted_images):
    
    image_files = [f for f in os.listdir(images) if os.path.isfile(os.path.join(images, f))]
    
    for subfolder in ['car', 'truck', 'motorcycle', 'bus']:
        os.makedirs(os.path.join(sorted_images, subfolder), exist_ok=True)

    for image_file in image_files:
        image_path = os.path.join(images, image_file)
        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)

        classes = ['car', 'truck', 'motorcycle', 'bus'] 
        predicted_class = classes[np.argmax(prediction)]

        destination_folder = os.path.join(sorted_images, predicted_class)
        shutil.move(image_path, destination_folder)

        print(f"Moved {image_file} to {predicted_class}")

classify_and_move('images', 'sorted_images')
