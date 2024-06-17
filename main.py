from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load a pre-trained AI model (e.g., a simple neural network for digit recognition)
model = tf.keras.models.load_model('path_to_your_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    # Convert the data into a numpy array
    input_data = np.array(data['input']).reshape(1, -1)
    # Make a prediction using the model
    prediction = model.predict(input_data)
    # Send the prediction back as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
