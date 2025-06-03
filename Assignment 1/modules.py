################################################################################
#
# BGU IEM Introduction to Deep Learning Course | 2024b
#
################################################################################

"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################


        self.in_features = in_features
        self.out_features = out_features
        self.input_layer = input_layer

        # Kaiming initialization for weights
        self.w = np.random.randn(out_features, in_features) * np.sqrt(2. / in_features)
        self.b = np.zeros(out_features)


        # Gradients
        self.grads = {
            'w': np.zeros_like(self.w),
            'b': np.zeros_like(self.b)
        }        

        


        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = x

        # Reshape input to (batch_size, num_features)
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, -1)

        # Check that the input and weights are compatible for matrix multiplication
        #print("x shape: ", x_reshaped.shape)
        #print("w shape: ", self.w.shape)
        assert x_reshaped.shape[1] == self.w.shape[1], f"Input ({x_reshaped.shape[1]}) and weights ({self.w.shape[1]}) are not compatible for matrix multiplication"

        # Compute the linear transformation
        out = x_reshaped @ self.w.T + self.b

    
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        dx = dout @ self.w
        #print("dout shape: ", dout.shape)
        #print("x shape: ", self.x.shape)
        #print("w shape: ", self.w.shape)
        
        # Reshape input x to match the shape of self.w before computing gradients
        batch_size = self.x.shape[0]
        x_reshaped = self.x.reshape(batch_size, -1)
        
        self.grads['w'] = dout.T @ x_reshaped
        self.grads['b'] = np.sum(dout, axis=0)

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = None





        #######################
        # END OF YOUR CODE    #
        #######################


class RELUModule(object):
    """
    RELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = x
        out = np.maximum(0, x)


        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        dx = dout * (self.x > 0)

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = None
        self.exp_x = None
        self.out = None
        self.cache = None

        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        # Compute softmax
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)

        # Compute softmax probabilities
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Store intermediate variables for backward pass
        self.softmax_out = out

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        # Compute Jacobian matrix
        softmax_out = self.softmax_out
        jacobian_matrix = np.einsum('ij,ik->ijk', -softmax_out, softmax_out)
        diag_indices = np.arange(softmax_out.shape[1])
        jacobian_matrix[:, diag_indices, diag_indices] += softmax_out

        # Compute dx using the chain rule
        dx = np.einsum('ijk,ik->ij', jacobian_matrix, dout)

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        self.softmax_out = None


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
      

        y = np.eye(10)[y]  # Convert integer labels to one-hot encoded format

        # Clip probabilities to avoid log(0) = -inf
        epsilon = 1e-10
        x_clipped = np.clip(x, epsilon, 1 - epsilon)

        # Compute cross-entropy loss
        out = -np.mean(np.sum( y * np.log(x_clipped), axis=1))

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        y = np.eye(10)[y]

        # Clip probabilities to avoid log(0) = -inf
        epsilon = 1e-10
        x_clipped = np.clip(x, epsilon, 1 - epsilon)

        # Compute gradient of the loss with respect to x
        dx = -y / x_clipped
        dx /= x.shape[0]  # Normalize by the batch size

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx