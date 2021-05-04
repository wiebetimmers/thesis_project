import torch
import torch.nn as nn


class Baseline_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, T, dataset=None):
        super(Baseline_RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.recurrent_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(input_size + hidden_size, output_size)

        self.f_0 = nn.Tanh()
        self.f_t = nn.ReLU(inplace=False)
        self.softmax = nn.LogSoftmax(dim=1)

        self.T = T

        # Initialize the weights
        self.input_hidden_layer.weight.data.uniform_(-0.01, 0.01)
        self.recurrent_layer.weight.data.uniform_(-0.01, 0.01)
        self.output_layer.weight.data.uniform_(-0.01, 0.01)

        self.dataset = dataset

    def forward(self, input):

        # Initialize hidden input & hidden layer
        initial_hidden = torch.zeros(input.size()[0], self.hidden_size)
        hidden = self.f_0(self.input_hidden_layer(initial_hidden))

        for t in range(self.T):
            if self.dataset == 'Digits':  # So we use the Digits dataset
                combined = torch.cat((input, hidden), 1)
                hidden = self.f_t(self.recurrent_layer(combined))

            elif self.dataset == 'MNIST':  # So we use the MNIST dataset

                # Squeeze the MNIST Tensor in 2 dimensions
                three_d_tensor = input.squeeze(1)
                two_d_tensor = three_d_tensor.contiguous().view(three_d_tensor.size()[0], -1)  # 28 * 28 pixels = 784

                combined = torch.cat((two_d_tensor, hidden), 1)
                hidden = self.f_t(self.recurrent_layer(combined))

        output = self.output_layer(combined)
        y = self.softmax(output)

        return y


class Reservoir_RNN(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, T, dataset=None):
        super(Reservoir_RNN, self).__init__()

        # Activation functions
        self.f_0 = nn.Tanh()
        self.f_t = nn.ReLU(inplace=False)
        self.f_y = nn.LogSoftmax(dim=1)

        # Amount of timesteps / recurrent layers
        self.T = T

        # Initialize the weights & layers
        self.initWeights(input_size, reservoir_size, output_size)
        self.initLayers(input_size, reservoir_size, output_size)

        # Either digits or mnist
        self.dataset = dataset

    def forward(self, input):

        used_input = input

        # Squeeze the used input in 2 dims if we use the MNIST dataset.
        if self.dataset == 'MNIST':
            three_d_tensor = used_input.squeeze(1)
            used_input = three_d_tensor.contiguous().view(three_d_tensor.size()[0], -1)  # 28 * 28 pixels = 784

        # Calculate c_0
        c = self.f_0(self.layer1(used_input))

        for t in range(self.T):
            # c_t =  f_t (W_r * c_t-1 + U * x_t)
            c = self.f_t(self.layer2(c) + self.layer3(used_input))

        # Calculate y = f_y ( W_out * c_t)
        y = self.f_y(self.layer4(c))

        return y

    def initWeights(self, input_size, reservoir_size, output_size):

        # Sample the initial weights from a uniform distribution - initialize the same as in the baseline model.
        self.W_in = nn.Parameter(data=torch.zeros(reservoir_size, input_size, requires_grad=False))
        self.W_in.data.uniform_(-0.01, 0.01)

        self.W_r = nn.Parameter(data=torch.zeros(reservoir_size, reservoir_size), requires_grad=False)
        self.W_r.data.uniform_(-0.01, 0.01)

        self.W_out = nn.Parameter(data=torch.zeros(output_size, reservoir_size), requires_grad=True)
        self.W_out.data.uniform_(-0.01, 0.01)

        self.U = nn.Parameter(data=torch.zeros(reservoir_size, input_size), requires_grad=False)
        self.U.data.uniform_(-0.01, 0.01)
        return

    def initLayers(self, input_size, reservoir_size, output_size):
        # Input layer
        self.layer1 = torch.nn.Linear(input_size, reservoir_size, bias=True)
        self.layer1.weight = self.W_in
        self.layer1.weight.requires_grad = False
        self.layer1.bias.requires_grad = False

        # Recurrent layer
        self.layer2 = torch.nn.Linear(reservoir_size, reservoir_size, bias=True)
        self.layer2.weight = self.W_r
        self.layer2.bias.requires_grad = False
        self.layer3 = torch.nn.Linear(input_size, reservoir_size, bias=True)
        self.layer3.weight = self.U
        self.layer3.bias.requires_grad = False

        # Output layer
        self.layer4 = torch.nn.Linear(reservoir_size, output_size, bias=True)
        self.layer4.weight = self.W_out
        self.layer4.bias.requires_grad = True
        return