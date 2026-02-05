## 🧠 Neural Networks for Handwritten Digit Recognition

This project explores the **mathematical foundations of neural networks**, with a focus on **deriving backpropagation from first principles**. The handwritten digit classifier is used as a concrete application to validate the mathematics.

Rather than treating the model as a black box, the project examines:
- Forward propagation through multiple layers
- **Backward propagation of gradients using the chain rule**
- Weight and bias updates during a single learning step

A **full iteration of backpropagation was computed manually by hand**, and the analytical results were verified using a Python/TensorFlow implementation.
📄 **Hand-derived backpropagation calculations:**  
[Google Sheets – Backpropagation Derivation](https://docs.google.com/spreadsheets/d/1Cq9TQVi4c1R4hMYBZ8yyiRYQwrktnMIA719n4Upulj8/edit?gid=0#gid=0)

The included code, trained models, and exported weight matrices primarily serve to:
- Validate gradient calculations
- Inspect learned parameters
- Connect mathematical theory to an implemented neural network

---

## 📂 Repository Contents  
- **`main.py`**: The main Python script for training and testing the neural network. 🐍  
- **`handwritten_digits.model`**: The trained neural network model. 🤖  
- **`handwritten_digits.model.keras`**: A version of the model saved in Keras format. 📦  
- **`layer_1_weights.csv`**: Weights of the first hidden layer. ⚖️  
- **`layer_1_biases.csv`**: Biases of the first hidden layer. ➕  
- **`layer_2_weights.csv`**: Weights of the second hidden layer. ⚖️  
- **`layer_2_biases.csv`**: Biases of the second hidden layer. ➕  
- **`layer_3_weights.csv`**: Weights of the output layer. ⚖️  
- **`layer_3_biases.csv`**: Biases of the output layer. ➕  
---

🎯 This project highlights the math and its role in powering artificial intelligence!  
