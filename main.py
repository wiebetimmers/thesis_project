import torch.optim as optim
from pytorch_model_summary import summary

import Networks as Net
import Operations as Ops
import Datasets as Data
import Parameters as P
from EA import EA

# Run baseline model
bl_model_digits = Net.Baseline_RNN(P.input_size_digits, P.n_hidden, P.n_labels, P.T, dataset = 'Digits')
optimizer_digits = optim.SGD([p for p in bl_model_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD, momentum=P.momentum_SGD)
trained_bl_digits = Ops.training(bl_model_digits, P.train_loader_digits, P.val_loader_digits, P.n_epochs, optimizer_digits, P.loss_function, P.max_loss_iter)

# Run reservoir ea

# Initialize population - train by backprop for a few epochs.
reservoir_set_digits = []
ea = EA(P.population_size, P.val_loader_digits, P.loss_function, P.input_size_digits, P.reservoir_size, P.n_labels)

for i in range(P.population_size):
    res_evo_digits = Net.Reservoir_RNN(P.input_size_digits, P.reservoir_size, P.n_labels, P.T, dataset='Digits')
    optimizer_evo_digits = optim.SGD([p for p in res_evo_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                                     momentum=P.momentum_SGD)
    trained_evo_digits = Ops.training(res_evo_digits, P.train_loader_digits, P.val_loader_digits, P.backprop_epochs,
                                  optimizer_evo_digits, P.loss_function, P.max_loss_iter)
    reservoir_set_digits.append(trained_evo_digits)

# Initialize the population
new_pop = reservoir_set_digits

# Perform ea steps
for i in range(P.generations):
    new_pop = ea.step(new_pop, P.mutate_opt, P.perturb_rate, P.select_opt, P.select_mech, P.offspring_ratio,
                      P.sample_dist, P.k_best, P.loss_function)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_pop_digits = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
elif P.select_opt == 'loss':
    best_pop_digits = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)


# Run RNN without evo
res_model_digits = Net.Reservoir_RNN(P.input_size_digits, P.reservoir_size, P.n_labels, P.T, dataset = 'Digits')
optimizer_digits = optim.SGD([p for p in res_model_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                             momentum=P.momentum_SGD)
trained_res_digits = Ops.training(res_model_digits, P.train_loader_digits, P.val_loader_digits, P.n_epochs,
                                  optimizer_digits, P.loss_function, P.max_loss_iter)

# Plot above plots in one plot
Ops.combined_plot_result(
            trained_bl_digits['epoch'],
            trained_bl_digits['loss_results'],
            trained_bl_digits['class_error_results'],
            trained_res_digits['loss_results'],
            trained_res_digits['class_error_results'],
            best_pop_digits[0]['loss_results'],
            best_pop_digits[0]['class_error_results'],
            border = P.backprop_epochs,
            label_bl = 'Baseline RNN',
            label_res = 'Reservoir RNN',
            label_evo = 'EA Reservoir RNN',
            title = 'Combined plot - fitness function is minimize classification error')

Ops.best_pop_plot(best_pop_digits,
              best_pop_digits[0],
              title='Final (best) population (size %s), individual performance' %(P.population_size))