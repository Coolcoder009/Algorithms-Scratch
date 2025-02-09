import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from layer import*
from loss import*
from activation import*


# Load MNIST Digits Dataset
digits = datasets.load_digits()
images = digits.images
labels = digits.target


# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, shuffle=False)
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)


# Convert the test labels into arrays of probabilities
new_labels = []
for label in y_train:
    probs = [0] * 10
    probs[label] = 1
    new_labels.append(probs)

num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]  # Optional for evaluation

# (Optional) Reshape one-hot labels if needed
y_train_reshaped = y_train_one_hot.reshape(y_train_one_hot.shape[0], -1)


def display_images(images, labels, title=None, predictions=None):
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8,8))
    fig.subplots_adjust(hspace=0.8)
    if title is not None: fig.suptitle(title, fontsize=20, fontweight="bold")

    for i in range(10):
        for j in range(10):
            axs[i][j].axis("off")
            axs[i][j].imshow(images[10*i+j].reshape((8,8)), cmap="Greys")
            # Display as model prediction/actual label
            if predictions is not None: axs[i][j].set_title(f"{predictions[10*i+j]}/{labels[10*i+j]}")
            else: axs[i][j].set_title(f"A:{labels[10*i+j]}")

    plt.show()


def convert(classifier_predictions):
    predictions = []
    for prediction in classifier_predictions:
        curr_pred = -1
        curr_prob = 0

        for i, val in enumerate(prediction[0]):
            if val > curr_prob:
                curr_pred = i
                curr_prob = val

        predictions.append(curr_pred)

    return predictions

if __name__ == "__main__":
    # Scale input data to avoid exploding gradients
    X_train_reshaped /= 16
    X_test_reshaped /= 16

    # Hyperparameters
    alpha = 1e-2
    batch_size = 32

    # MNIST digits classifier
    # Use LayerList to initialize neural network model
    # Add Layer objects to model either in Model instantiation or using Model.append() method
    classifier = Model(Layer(64, 28),
                           Layer(28, 10, activation_function = Softmax()))


    # Train using Model.fit() method
    classifier.fit(X_train_reshaped, y_train_reshaped, 1000, alpha, batch_size, categorical_cross_entropy_loss)

    # Evaluate using LayerList.predict() method
    display_images(X_test, y_test, title="NumpyNN Predictions", predictions=convert(classifier.predict(X_test_reshaped)))