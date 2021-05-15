import datetime  # Keep track of execution time.
import sys
begin_time = datetime.datetime.now()

import torch
import torch.optim as optim
from pytorch_model_summary import summary
import Networks as Net
import Operations as Ops
import Datasets as Data
import Parameters as P
import pickle
from EA import EA

# Load data
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)

# -----------------------------------------------------------------------------------------------------------
# Run reservoir EA model

# Initialize population - train by backprop for a few epochs.
reservoir_set_digits = []
ea = EA(P.population_size, val_loader_digits, P.loss_function, P.input_size_digits, P.reservoir_size, P.n_labels)

for i in range(P.population_size):
    res_evo_digits = Net.Reservoir_RNN(P.input_size_digits, P.reservoir_size, P.n_labels, P.T, dataset='Digits')
    optimizer_evo_digits = optim.SGD([p for p in res_evo_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                                     momentum=P.momentum_SGD)
    trained_evo_digits = Ops.training(res_evo_digits, train_loader_digits, val_loader_digits, P.backprop_epochs,
                                  optimizer_evo_digits, P.loss_function, P.max_loss_iter)
    reservoir_set_digits.append(trained_evo_digits)

# Initialize the population
new_pop = reservoir_set_digits

# Perform ea steps
for i in range(P.generations):
    new_pop = ea.step(new_pop, P.mutate_opt, P.mutate_bias, P.perturb_rate, P.select_opt, P.select_mech, P.offspring_ratio,
                      P.sample_dist, P.k_best, P.loss_function, P.mu, P.sigma)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_pop_digits = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
elif P.select_opt == 'loss':
    best_pop_digits = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)

# Save model and results dict
ea_reservoir_model = open('models/EA_reservoir_model.pkl', 'wb')
pickle.dump(best_pop_digits, ea_reservoir_model)
ea_reservoir_model.close()

# -----------------------------------------------------------------------------------------------------------
# Run baseline model
bl_model_digits = Net.Baseline_RNN(P.input_size_digits, P.n_hidden, P.n_labels, P.T, dataset = 'Digits')
optimizer_digits = optim.SGD([p for p in bl_model_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD, momentum=P.momentum_SGD)
trained_bl_digits = Ops.training(bl_model_digits, train_loader_digits, val_loader_digits, P.n_epochs, optimizer_digits, P.loss_function, P.max_loss_iter)

# Save model and results dict
baseline_model = open('models/baseline_model.pkl', 'wb')
pickle.dump(trained_bl_digits, baseline_model)
baseline_model.close()

# -----------------------------------------------------------------------------------------------------------

# Run RNN without evo
res_model_digits = Net.Reservoir_RNN(P.input_size_digits, P.reservoir_size, P.n_labels, P.T, dataset = 'Digits')
optimizer_digits = optim.SGD([p for p in res_model_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                             momentum=P.momentum_SGD)
trained_res_digits = Ops.training(res_model_digits, train_loader_digits, val_loader_digits, P.n_epochs,
                                  optimizer_digits, P.loss_function, P.max_loss_iter)

# Save model and results dict
reservoir_model_no_evo = open('models/reservoir_model_no_evo.pkl', 'wb')
pickle.dump(trained_res_digits, reservoir_model_no_evo)
reservoir_model_no_evo.close()

# ----------------------------------------------------------------------------------------------------------

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

print('\nExecution time was: (hours:minute:seconds:microseconds) %s \n' %exc_time)

