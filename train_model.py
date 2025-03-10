import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv('furniture_dataset.csv')
df.dropna(inplace=True)

# Prepare input (room and furniture sizes) and output (optimal positions)
X = df[['room_width', 'room_height', 'furniture_width', 'furniture_height']].values
y = df[['x_position', 'y_position']].values

# Normalize input/output
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Define model
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(2, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# Save model and scalers
model.save("furniture_model.h5")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("Model and scalers saved successfully!")
