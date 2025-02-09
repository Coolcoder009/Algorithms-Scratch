import numpy as np


# if x < = 0 , then return 0 else 1
class Relu:
    def __call__(self,pre_activated_output):
        return np.maximum(pre_activated_output,0)

    def derivative(self,pre_activated_output,grad_so_far):
        return np.where(pre_activated_output<=0,0,1)*grad_so_far

# 1/(1+e^(-x))
class Sigmoid:
    def __call__(self,pre_activated_output):
        pre_activated_output = np.clip(pre_activated_output,-1000,1000)
        return 1/(1+np.exp(-pre_activated_output))

    def derivative(self,pre_activated_output,grad_so_far):
        return (1/(1+np.exp(-pre_activated_output))) * (1-(1/(1+np.exp(-pre_activated_output)))) * grad_so_far

# Suppose pre_activated_output = [1,2,3] and if Relu is = [1,2,0], max in Relu is = 2
# So e^(1-2), e^(2-2), e^(3-(-2)) --> expected = [e^-1, e^0, e^5] // subtraction avoids max exponents
# denominator = (e^-1 + e^0 + e^5)
class Softmax():
    def __call__(self, output):
        exp_shifted = np.exp(output - np.max(output, axis=1, keepdims=True))
        denominator = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / denominator

    def derivative(self, output, grad_so_far):
        output = self(output)  # Get activated outputs for formulae
        batch_size, n_classes = output.shape

        # For 1 example, the jacobian is of size NxN, so for B batches, it is BxNxN
        jacobian = np.zeros((batch_size, n_classes, n_classes))

        for b in range(batch_size):
            out = output[b].reshape(-1, 1)  # Flatten output to be an Nx1 matrix
            jacobian[b] = np.diagflat(out) - np.dot(out, out.T)  # Create Jacobian for particular example
        return np.einsum('bij,bj->bi', jacobian, grad_so_far)  # Efficient batch-wise dot product using Einstein summation notation
