from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import uuid
from tensorflow.keras.losses import MeanSquaredError
from keras.saving import register_keras_serializable
import joblib  # For loading scalers

# Register loss function to fix serialization issue
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load trained model with custom objects
model = tf.keras.models.load_model("furniture_model.h5", custom_objects={"mse": MeanSquaredError()})

# Load MinMaxScaler objects
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        room_width = int(request.form['room_width'])
        room_height = int(request.form['room_height'])
        furniture_width = int(request.form['furniture_width'])
        furniture_height = int(request.form['furniture_height'])

        # Validate input values
        if min(room_width, room_height, furniture_width, furniture_height) <= 0:
            return render_template("index.html", error="All dimensions must be positive values.")
        
        if furniture_width > room_width or furniture_height > room_height:
            return render_template("index.html", error="Furniture cannot be larger than the room.")

        # Prepare input data and normalize it using pre-trained scalers
        input_data = np.array([[room_width, room_height, furniture_width, furniture_height]])
        input_data = scaler_X.transform(input_data)  # Normalize input

        # Predict optimal furniture placement
        predicted_position = model.predict(input_data)
        predicted_position = scaler_y.inverse_transform(predicted_position)  # Inverse transform output

        # Extract predicted x and y positions
        x_opt, y_opt = predicted_position[0]

        # Ensure furniture stays within room boundaries
        x_opt = max(0, min(x_opt, room_width - furniture_width))
        y_opt = max(0, min(y_opt, room_height - furniture_height))

        # Generate visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, room_width)
        ax.set_ylim(0, room_height)

        # Draw room
        ax.add_patch(plt.Rectangle((0, 0), room_width, room_height, 
                                   color='lightgray', alpha=0.5, 
                                   label='Room'))

        # Draw furniture
        ax.add_patch(plt.Rectangle((x_opt, y_opt), furniture_width, furniture_height, 
                                   color='blue', alpha=0.7, 
                                   label='Furniture'))

        ax.set_title("Optimized Furniture Placement")
        plt.xlabel("Width (units)")
        plt.ylabel("Height (units)")
        plt.grid(True)
        plt.legend()

        # Save image with a unique filename (avoid caching issues)
        img_filename = f"layout_{uuid.uuid4().hex[:8]}.png"
        img_path = os.path.join("static", img_filename)
        plt.savefig(img_path)
        plt.close()

        return render_template("index.html", 
                               x=int(x_opt), 
                               y=int(y_opt), 
                               room_width=room_width, 
                               room_height=room_height,
                               furniture_width=furniture_width, 
                               furniture_height=furniture_height,
                               image_url=img_path)

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
