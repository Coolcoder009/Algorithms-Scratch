import numpy as np
from activation import *
from random import shuffle

# A layer of the model
class Layer:
    def __init__(self, input_size, output_size, bias=False, activation_function=None):
        self.alpha = 1e-3
        self.bias = bias
        self.activation_layer = activation_function

        # This instantiates numpy array with random values with the size (i/p size, o/p size)
        self.weights = np.random.rand(input_size, output_size)
        if bias:
            self.weights = np.vstack((self.weights, np.ones((1,self.weights.shape[1]))))

        self.current_inputs = None
        self.update_matrix = None
        self.pre_activated_output = None

    # Forward pass
    def __call__(self, layer_inputs):  # layer_input = i/p size , batch size
        if self.bias:  # bias is stacked vertically onto weights so if there is one, we need to handle so we apply ones
            layer_inputs = np.hstack((layer_inputs, np.ones((layer_inputs.shape[0], 1))))

        self.current_inputs = np.copy(layer_inputs)
        layer_outputs = layer_inputs @ self.weights

        if self.activation_layer:
            self.pre_activated_output = np.copy(layer_outputs)
            layer_outputs = self.activation_layer(layer_outputs)
        return layer_outputs

    # Backward pass
    def back(self, ret):
        if self.activation_layer:  # Optional activation layer
            ret = self.activation_layer.derivative(self.pre_activated_output, ret)

        self.update_matrix = self.current_inputs.T @ ret
        new_ret = ret @ self.weights.T

        if self.bias:  # Remove bias column if needed
            new_ret = new_ret[:, :-1]
        return new_ret

    # Update layer weights
    def update(self):
        self.weights -= self.alpha * self.update_matrix
        self.pre_activated_output = None
        self.current_inputs = None
        self.update_matrix = None

    # Update learning rate
    def set_alpha(self, new_alpha):
        self.alpha = new_alpha



# Model with layers
class Model:
    def __init__(self, *layers):
        self.model = list(layers)

    def append_layers(self, *layers):
        for layer in layers:
            self.model.append(layer)

    def __call__(self, model_input):
        intermediate_results = model_input

        for layer in self.model:
            intermediate_results = layer(intermediate_results)

        return intermediate_results

    def back(self, error):
        for layer in self.model[::-1]:
            error = layer.back(error)

    def step(self):
        for layer in self.model:
             layer.update()

    def predict(self, inputs):
        pred=[]

        for i in inputs:
            pred.append(self(np.expand_dims(i, axis=0)))

        return pred

    @staticmethod
    def batch(input_data, expected, batch_size):
        data = input_data.shape[0]
        indices = [i for i in range(data)]
        shuffle(indices)

        batch_input, batch_expected = [], []
        for i in range(data//batch_size):
            batch_inp, batch_exp = [], []

            for j in range(batch_size):
                batch_inp.append(input_data[indices[batch_size*i+j]])
                batch_exp.append(expected[indices[batch_size*i+j]])
            batch_input.append(batch_inp)
            batch_expected.append(batch_exp)

        return np.array(batch_input), np.array(batch_expected)

    def set_alpha(self, new_alpha):
        for layer in self.model:
            layer.set_alpha(new_alpha)

    def fit(self, input_data, expected, epochs, alpha, batch_size, loss_deriv_func):
        if len(self.model) == 0:
            return

        total_iter = epochs
        self.set_alpha(alpha)

        while epochs:
            epochs -= 1
            batched_data, batched_expected = Model.batch(input_data, expected, batch_size)

            for idx, data_batch in enumerate(batched_data):
                output = self(data_batch)
                self.back(loss_deriv_func(output, batched_expected[idx]))
                self.step()

            if epochs == total_iter // 10:
                # Reducing learning rate to hone in on minima of loss function
                alpha /= 10
                self.set_alpha(alpha)
