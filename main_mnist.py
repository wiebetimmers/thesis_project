import datetime  # Keep track of execution time.
begin_time = datetime.datetime.now()

import sys
import torch
import torch.optim as optim
from pytorch_model_summary import summary
import Networks as Net
import Operations as Ops
import Datasets as Data
import Parameters as P
from EA import EA

# Run reservoir ea
train_loader_mnist, val_loader_mnist, test_loader_mnist = Data.get_mnist_loaders(P.batch_size)

# Initialize population - train by backprop for a few epochs.
reservoir_set_mnist = []
ea = EA(P.population_size, val_loader_mnist, P.loss_function, P.input_size_mnist, P.reservoir_size, P.n_labels)

for i in range(P.population_size):
    res_evo_mnist = Net.Reservoir_RNN(P.input_size_mnist, P.reservoir_size, P.n_labels, P.T, dataset='MNIST')
    optimizer_evo_mnist = optim.SGD([p for p in res_evo_mnist.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                                     momentum=P.momentum_SGD)
    trained_evo_mnist = Ops.training(res_evo_mnist, train_loader_mnist, val_loader_mnist, P.backprop_epochs,
                                  optimizer_evo_mnist, P.loss_function, P.max_loss_iter)
    reservoir_set_mnist.append(trained_evo_mnist)

# Initialize the population
new_pop = reservoir_set_mnist

# Perform ea steps
for i in range(P.generations):
    new_pop = ea.step(new_pop, P.mutate_opt, P.mutate_bias, P.perturb_rate, P.select_opt, P.select_mech, P.offspring_ratio,
                      P.sample_dist, P.k_best, P.loss_function)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_pop_mnist = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
elif P.select_opt == 'loss':
    best_pop_mnist = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)

# Run baseline model
bl_model_mnist = Net.Baseline_RNN(P.input_size_mnist, P.n_hidden, P.n_labels, P.T, dataset = 'MNIST')
optimizer_mnist = optim.SGD([p for p in bl_model_mnist.parameters() if p.requires_grad == True], lr=P.lr_SGD, momentum=P.momentum_SGD)
trained_bl_mnist = Ops.training(bl_model_mnist, train_loader_mnist, val_loader_mnist, P.n_epochs, optimizer_mnist, P.loss_function, P.max_loss_iter)

# Run RNN without evo
res_model_mnist= Net.Reservoir_RNN(P.input_size_mnist, P.reservoir_size, P.n_labels, P.T, dataset = 'MNIST')
optimizer_mnist = optim.SGD([p for p in res_model_mnist.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                             momentum=P.momentum_SGD)
trained_res_mnist = Ops.training(res_model_mnist, train_loader_mnist, val_loader_mnist, P.n_epochs,
                                  optimizer_mnist, P.loss_function, P.max_loss_iter)

# Plot above plots in one plot
Ops.combined_plot_result(
            trained_bl_mnist['epoch'],
            trained_bl_mnist['loss_results'],
            trained_bl_mnist['class_error_results'],
            trained_res_mnist['loss_results'],
            trained_res_mnist['class_error_results'],
            best_pop_mnist[0]['loss_results'],
            best_pop_mnist[0]['class_error_results'],
            border = P.backprop_epochs,
            label_bl = 'Baseline RNN',
            label_res = 'Reservoir RNN',
            label_evo = 'EA Reservoir RNN',
            title = 'Combined plot MNIST - fitness function is minimize classification error')

Ops.best_pop_plot(best_pop_mnist,
              best_pop_mnist[0],
              title='Final (best) population (size %s) MNIST, individual performance' %(P.population_size))


sys.stdout = open("test_results_mnist.txt", "w")
# Network / Learning parameters
print('Network parameters:\n'
      'reservoir size: %s, \n'
      'n_hidden: %s, \n'
      'learning rate: %s, \n'
      'momentum sgd: %s, \n'
      'backprop epochs: %s, \n'
      'T: %s, \n'
      'loss_function: %s \n' % (P.reservoir_size, P.n_hidden, P.lr_SGD, P.momentum_SGD,
                             P.backprop_epochs, P.T, P.loss_function))
print('EA parameters: \n'
      ' pop size: %s,\n'
      'generations: %s,\n'
      'mutate opt: %s,\n'
      'perturb rate: %s,\n'
      'mutate_bias: %s\n'
      'sample_dist: %s\n'
      'select opt: %s\n'
      'select mech: %s\n'
      'k_best: %s\n'
      'offspring ratio: %s\n'
      'n epochs: %s\n' % (P.population_size,P.generations , P.mutate_opt, P.perturb_rate,
P.mutate_bias, P.sample_dist,      P.select_opt,  P.select_mech,  P.k_best,  P.offspring_ratio, P.n_epochs))

test_result_mnist = Ops.evaluation(test_loader_mnist, trained_res_mnist['model'], 'Final score MNIST on test set - only output train', P.loss_function)
test_result_mnist2 = Ops.evaluation(test_loader_mnist, trained_bl_mnist['model'], 'Final score MNIST on test set- baseline', P.loss_function)
test_result_mnist3 = Ops.evaluation(test_loader_mnist, best_pop_mnist[0]['model'], 'Final score MNIST on test set- with evolution', P.loss_function)

# Baseline RNN model
print(summary(bl_model_mnist, torch.zeros(1, 64), show_input=True, show_hierarchical=False))
# Reservoir RNN model
print(summary(res_model_mnist, torch.zeros(1, 64), show_input=True, show_hierarchical=False))

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

sys.stdout.close()

print('Execution time was: (hours:minute:seconds:microseconds) %s ' %exc_time)
