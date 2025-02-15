Neural Network from Scratch (Using Python Classes & Objects)
Overview
This project implements a simple feedforward neural network from scratch using OOP concepts in Python. It supports:
✅ Multiple layers with customizable input/output sizes.
✅ Activation functions (e.g., Softmax for classification).
✅ Forward propagation, backpropagation, and weight updates.
✅ Mini-batch training for efficient learning.

Project Structure
Model – Stores and manages layers, handles forward and backward passes.
Layer – Represents a fully connected layer with weights, biases, and activations.
Softmax – Activation function used in the final output layer.
categorical_cross_entropy_loss – Loss function used for classification tasks.
How It Works (Training Flow)
1️⃣ Initialize the Model

python
Copy
Edit
classifier = Model(
    Layer(64, 28),  
    Layer(28, 10, activation_function=Softmax())  
)
2️⃣ Train the Model

python
Copy
Edit
classifier.fit(X_train, y_train, epochs=1000, alpha=0.01, batch_size=32, loss_deriv_func=categorical_cross_entropy_loss_derivative)
3️⃣ Make Predictions

python
Copy
Edit
predictions = classifier.predict(X_test)
Flow Diagram

![Alt Text](https://github.com/Coolcoder009/NeuralNetworks-Scratch/blob/main/Flow/Neural%20Network.png?raw=true)
