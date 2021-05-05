import datetime  # Keep track of execution time.
begin_time = datetime.datetime.now()

import torch
import torch.optim as optim
from pytorch_model_summary import summary
import Networks as Net
import Operations as Ops
import Datasets as Data
import Parameters as P
from EA import EA

# Run reservoir ea
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)

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
                      P.sample_dist, P.k_best, P.loss_function)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_pop_digits = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
elif P.select_opt == 'loss':
    best_pop_digits = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)


# Run baseline model
bl_model_digits = Net.Baseline_RNN(P.input_size_digits, P.n_hidden, P.n_labels, P.T, dataset = 'Digits')
optimizer_digits = optim.SGD([p for p in bl_model_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD, momentum=P.momentum_SGD)
trained_bl_digits = Ops.training(bl_model_digits, train_loader_digits, val_loader_digits, P.n_epochs, optimizer_digits, P.loss_function, P.max_loss_iter)

# Run RNN without evo
res_model_digits = Net.Reservoir_RNN(P.input_size_digits, P.reservoir_size, P.n_labels, P.T, dataset = 'Digits')
optimizer_digits = optim.SGD([p for p in res_model_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                             momentum=P.momentum_SGD)
trained_res_digits = Ops.training(res_model_digits, train_loader_digits, val_loader_digits, P.n_epochs,
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
            title = 'Combined plot Digits - fitness function is minimize classification error')

Ops.best_pop_plot(best_pop_digits,
              best_pop_digits[0],
              title='Final (best) population (size %s) Digits, individual performance' %(P.population_size))

test_result_digits = Ops.evaluation(test_loader_digits, trained_res_digits['model'], 'Final score Digits on test set - only output train', P.loss_function)
test_result_digits2 = Ops.evaluation(test_loader_digits, trained_bl_digits['model'], 'Final score Digits on test set- baseline', P.loss_function)
test_result_digits3 = Ops.evaluation(test_loader_digits, best_pop_digits[0]['model'], 'Final score Digits on test set- with evolution', P.loss_function)

# Baseline RNN model
# print(summary(bl_model_digits, torch.zeros(1, 64), show_input=True, show_hierarchical=False))
# Reservoir RNN model
# print(summary(res_model_digits, torch.zeros(1, 64), show_input=True, show_hierarchical=False))

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

print('Execution time was: (hours:minute:seconds:microseconds) %s ' %exc_time)
