import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from keras.models import load_model
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

app = Flask(__name__)

# Load saved model
model = load_model('model_for_10snakes.h5')
app.config.from_pyfile('settings.py')


# Define endpoint for receiving image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="Please try again. The image doesn't exist.")

    # Get image from POST request
    file = request.files.get('file')
    img_bytes = file.read()
    img_path = "./uploadImages/test.jpg"
    with open(img_path, "wb") as img:
        img.write(img_bytes)

    # Load image and convert to NumPy array
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.

    # Make prediction

    prediction = model.predict(img_array)

    # Convert prediction to class label and confidence level
    predicted_class = np.argmax(prediction)
    confidence_level = np.max(prediction)

    # Return JSON response

    if confidence_level >= 0.65:
        response = {'predicted_class': int(predicted_class), 'confidence_level': float(confidence_level)}
        print(response)
        return jsonify(response)
    else:
        response = {'predicted_class': -2, 'confidence_level': None}
        print(response)
        return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
