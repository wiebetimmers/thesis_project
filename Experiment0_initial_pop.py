import torch.optim as optim
import Networks as Net
import Operations as Ops
import Datasets as Data
import Parameters as P
import pickle

# Load data
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)

# Start with the same backpropped population for each distribution - so we run this file once for experiment 1 and 2.
reservoir_set_digits = []

for i in range(P.population_size):
    res_evo_digits = Net.Reservoir_RNN(P.input_size_digits, P.reservoir_size, P.n_labels, P.T, dataset='Digits')
    optimizer_evo_digits = optim.SGD([p for p in res_evo_digits.parameters() if p.requires_grad], lr=P.lr_SGD,
                                     momentum=P.momentum_SGD)
    trained_evo_digits = Ops.training(res_evo_digits, train_loader_digits, val_loader_digits, P.backprop_epochs,
                                      optimizer_evo_digits, P.loss_function, P.max_loss_iter)
    reservoir_set_digits.append(trained_evo_digits)

reservoir_model = open('models/digits_reservoir_model_RP_start.pkl', 'wb')
pickle.dump(reservoir_set_digits, reservoir_model)
reservoir_model.close()
