import numpy as np
from activation import *
# A layer of the model
class Layer:
    def __init__(self,input_size,output_size,alpha,batch_size,bias=False,activation_function=None):
        self.input_size=input_size
        self.output_size=output_size
        self.alpha=alpha
        self.batch_size=batch_size
        self.bias=bias
        self.activation_function=activation_function

        # This instantiates numpy array with random values with the size (i/p size, o/p size)
        self.weights=np.random.rand(self.input_size,self.output_size)
        if bias:
            self.weights=np.vstack(self.weights,np.random.rand((1,self.output_size)))


        self.current_inputs=None
        self.update_matrix=None
        self.pre_activated_output=None

    def get_batch_size(self):
        return self.batch_size
    
    def set_alpha(self,new_alpha):
        self.alpha=new_alpha

    def __call__(self,layer_inputs): # layer_input = i/p size , batch size
        if self.bias: # bias is stacked vertically onto weights so if there is one, we need to handle so we apply ones
            layer_inputs = np.hstack(layer_inputs,np.ones((self.batch_size,1)))

        self.current_inputs=np.copy(layer_inputs)
        layer_outputs = layer_inputs @ self.weights

        if self.activation_function is not None:
            self.pre_activated_output=np.copy(layer_outputs)
            layer_outputs=self.activation_function(layer_outputs)

        return layer_outputs

    def back(self, ret):
        if self.activation_function:  # Optional activation layer
            ret = self.activation_function.derivative(self.pre_activated_output, ret)

        self.update_matrix = self.current_inputs.T @ ret
        new_ret = ret @ self.weights.T

        if self.bias:  # Remove bias column if needed
            new_ret = new_ret[:, :-1]

        return new_ret

    def update(self):
        self.weights -= self.alpha * self.update_matrix
        self.pre_activated_output = None
        self.current_inputs = None
        self.update_matrix = None

# Model with layers
class Model:
    def __init__(self,*layers):
        self.model=list(layers)

    def append_layers(self,*layers):
        for layer in layers:
            self.model.append(layer)

    def set_alpha(self,new_alpha):
        for layer in self.model:
            layer.set_alpha(new_alpha)

    def __call__(self,model_input):
        intermediate_results=model_input

        for layer in self.model:
            intermediate_results=layer(intermediate_results)

        return intermediate_results

    def back(self,error):
        for layer in self.model[::-1]:
            error=layer.back(error)

    def step(self):
        for layer in self.model:
             layer.update()


    @staticmethod
    def batch(input_data,expected,batch_size):
        data=input_data.shape[0]
        indices=[i for i in range(data)]
        shuffle(indices)

        batch_input,batch_expected=[],[]
        for i in range(data//batch_size):
            batch_inp,batch_exp=[],[]

            for j in range(batch_size):
                batch_inp.append(input_data[indices[batch_size*i+j]])
                batch_exp.append(input_data[indices[batch_size*i+j]])
            batch_input.append(np.array(batch_inp))
            batch_expected.append(np.array(batch_exp))

        return np.array(batch_input),np.array(batch_expected)

    def fit(self,input_data,expected,epochs,alpha,batch_size,loss_deriv_func):
        self.set_alpha(alpha)
        prev_update=1
        for e in range(epochs):
            batch_input,batch_expected= Model.batch(input_data,expected,batch_size)
            for i in range(len(batch_input)):
                model_op=self(batch_input[i])
                self.back(loss_deriv_func(model_op,batch_expected[i]))
                self.step()

            if e==10*prev_update:
                alpha/=10
                self.set_alpha(alpha)
                prev_update=e

    def predict(self,inputs):
        pred=[]
        for i in inputs:
            pred.append(self(np.expand_dims(i,axis=0)))

        return pred

if __name__=="__main__":
    model= Model(Layer(1,1,0.1,1),Layer(1,2,0.1,1,activation_function=Softmax()))
    inp=np.array([[1]])
    print(model(inp))