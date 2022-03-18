from layer_base import BaseLayer


class FlattenLayer(BaseLayer):
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1, -1))
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)
