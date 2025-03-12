# 🏠 AI-Based Furniture Placement Optimizer

This project leverages **Artificial Intelligence (AI)** to **optimize furniture placement** within a room. Using **TensorFlow**, a **Neural Network (MLP)** predicts the best position for furniture based on room size, furniture dimensions, and spatial constraints.

---

## 🚀 Project Overview
### **🛠️ Problem Statement**
Arranging furniture efficiently in a limited space is challenging. People often struggle with finding the best placement that:
- Maximizes available space
- Avoids overcrowding or poor accessibility
- Maintains functional aesthetics  

This AI-powered optimizer **automates the process**, providing the best possible arrangement for a given furniture size and room dimensions.

### **🤖 How It Works**
1. **User Inputs:** Room width & height, furniture dimensions  
2. **AI Model Predicts:** The most optimal `(x, y)` coordinates  
3. **Visualization:** The placement is **displayed as a 2D layout**  
4. **Interactive UI:** Users enter values and get instant results  

---

## 🧠 AI Model Details
### **📌 Model Type**
- **Architecture:** Multi-Layer Perceptron (**MLP**)  
- **Input Features:** `[Room Width, Room Height, Furniture Width, Furniture Height]`  
- **Output:** `[Optimal X-Position, Optimal Y-Position]`  
- **Activation Functions:**  
  - **Hidden Layers:** ReLU  
  - **Output Layer:** Linear  

### **📊 Dataset**
- **Synthetic Dataset Generated** (2,00,000 samples)  
- **Data Columns:** (6 columns)
  
- The dataset ensures **realistic furniture placement** within the room's bounds.

### **🎯 Training Process**
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Epochs:** 50  
- **Batch Size:** 32  
- **Train-Test Split:** 90%-10%  

---

## 💻 Installation & Setup
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/naveennekkanti1/Furniture-Placement-AI.git
cd Furniture-Placement-AI
pip install -r requirements.txt
python train_model.py
python app.py


