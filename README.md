<h2>Neural Network from Scratch (Using Python Classes & Objects)</h2><br>

This project implements a simple feedforward neural network from scratch using OOP concepts in Python.<br> It supports:<br>
✅ Multiple layers with customizable input/output sizes.<br>
✅ Activation functions (e.g., Softmax for classification).<br>
✅ Forward propagation, backpropagation, and weight updates.<br>
✅ Mini-batch training for efficient learning.<br>

Project Structure
Model – Stores and manages layers, handles forward and backward passes.
Layer – Represents a fully connected layer with weights, biases, and activations.
Softmax – Activation function used in the final output layer.
categorical_cross_entropy_loss – Loss function used for classification tasks.
How It Works (Training Flow) <br>
1️⃣ Initialize the Model

classifier = Model(
    Layer(64, 28),  
    Layer(28, 10, activation_function=Softmax())) <br>
2️⃣ Train the Model

classifier.fit(X_train, y_train, epochs=1000, alpha=0.01, batch_size=32, loss_deriv_func=categorical_cross_entropy_loss_derivative) <br>
3️⃣ Make Predictions

predictions = classifier.predict(X_test) <br>
Flow Diagram

![Alt Text](https://github.com/Coolcoder009/NeuralNetworks-Scratch/blob/main/Flow/Neural%20Network.png?raw=true)
