from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load model 
model = load_model('/Users/remiore/Desktop/ImageClassification/models/imagemodel.h5')

# Define class labels
class_labels = ['Sad', 'Happy']  

# Preprocessing function
def preprocess_image(img):
    img = img.resize((256, 256))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
    img_array /= 255.0  # Normalizing pixel values to [0, 1]
    return img_array

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Save the uploaded image
        imagefile = request.files.get('imagefile')
        if not imagefile:
            return render_template('index.html', prediction="No image uploaded!")

        # Save image to a temporary folder
        image_path = "./images/"
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_file_path = os.path.join(image_path, imagefile.filename)
        imagefile.save(image_file_path)

        # Load and preprocess the image
        img = Image.open(image_file_path).convert('RGB')  # Convert to RGB if necessary
        img_array = preprocess_image(img)

        # Make a prediction
        predictions = model.predict(img_array)
        print(f"Predictions: {predictions}")



        max_index = np.argmax(predictions)  # Index of highest probability
        min_index = np.argmin(predictions)


        confidence = predictions[0][max_index] * 100
        if round(predictions[0][0]) == 1:
            classification = f"{class_labels[0]} (Confidence level: {confidence:.2f}%)"

        else:
            confidence = 100 - (predictions[0][min_index] * 100)
            classification = f"{class_labels[1]} (Confidence level: {confidence:.2f}%)"

        return render_template('index.html', prediction=classification)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(port=2000, debug=True)







