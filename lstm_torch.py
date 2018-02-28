import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TorchLSTM(object):
	def __init__(self, input_size, hidden_layers, hidden_layer_sizes, output_size):
		self.input_size = input_size
		self.hidden_layers = hidden_layers
		self.hidden_layer_sizes = hidden_layer_sizes
		self.output_size = output_size

		self.create_model()

	def create_model(self):
		self.model = nn.LSTM(self.input_size, self.ouput_size)
		for i in range(0, self.hidden_layers):
			size = self.hidden_layer_sizes[i]


if __name__ == '__main__':
	data = [[1, 2, 3], [2, 3, 3], [1, 3, 4]]
	output = [1, 2, 3]
	hidden_layers = 2
	hidden_layer_sizes = [15, 15]
	output_size = 1
	input_size = 3

	model = TorchLSTM(input_size, hidden_layers, hidden_layer_sizes, output_size)
