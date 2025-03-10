import numpy as np
import pandas as pd

# Generate synthetic dataset
num_samples = 5000
room_sizes = np.random.randint(8, 20, size=(num_samples, 2))  # Room width, height
furniture_sizes = np.random.randint(2, 6, size=(num_samples, 2))  # Furniture width, height

# Optimal placement (random but ensuring no out-of-bounds issues)
x_positions = np.random.randint(0, room_sizes[:, 0] - furniture_sizes[:, 0])
y_positions = np.random.randint(0, room_sizes[:, 1] - furniture_sizes[:, 1])

# Create DataFrame
df = pd.DataFrame({
    'room_width': room_sizes[:, 0], 'room_height': room_sizes[:, 1],
    'furniture_width': furniture_sizes[:, 0], 'furniture_height': furniture_sizes[:, 1],
    'x_position': x_positions, 'y_position': y_positions
})

# Save dataset
df.to_csv('furniture_dataset.csv', index=False)
print("Dataset generated and saved as furniture_dataset.csv!")
