import datetime
import torch.optim as optim
import torch
from pytorch_model_summary import summary
import Networks as Net
import Operations as Ops
import Datasets as Data
import sys
import Parameters as P
import pickle
from EA import EA

# Load data
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)

# -----------------------------------------------------------------------------------------------------------
# Run reservoir EA model
# -----------------------------------------------------------------------------------------------------------

# Initialize population - train by backprop for a few epochs.
reservoir_set_digits = []
ea = EA(P.population_size, val_loader_digits, P.loss_function, P.input_size_digits, P.reservoir_size, P.n_labels)

for i in range(P.population_size):
    res_evo_digits = Net.Reservoir_RNN(P.input_size_digits, P.reservoir_size, P.n_labels, P.T, dataset='Digits')
    optimizer_evo_digits = optim.SGD([p for p in res_evo_digits.parameters() if p.requires_grad], lr=P.lr_SGD,
                                     momentum=P.momentum_SGD)
    trained_evo_digits = Ops.training(res_evo_digits, train_loader_digits, val_loader_digits, P.backprop_epochs,
                                      optimizer_evo_digits, P.loss_function, P.max_loss_iter)
    reservoir_set_digits.append(trained_evo_digits)

# Initialize the population - store the backpropped 5 epochs so we can continue to use the same for diff experiments
# Comment this part out if there already is a sampled begin population
initial_model = open('models/exp3/digits_initial_model_start_pop_%s.pkl' % P.population_size, 'wb')
pickle.dump(reservoir_set_digits, initial_model)
initial_model.close()

