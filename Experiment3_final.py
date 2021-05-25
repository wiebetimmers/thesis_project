import datetime  # Keep track of execution time.
begin_time = datetime.datetime.now()

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
    optimizer_evo_digits = optim.SGD([p for p in res_evo_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                                     momentum=P.momentum_SGD)
    trained_evo_digits = Ops.training(res_evo_digits, train_loader_digits, val_loader_digits, P.backprop_epochs,
                                  optimizer_evo_digits, P.loss_function, P.max_loss_iter)
    reservoir_set_digits.append(trained_evo_digits)

# Initialize the population
new_pop = reservoir_set_digits

# Perform ea steps
for i in range(P.generations):
    new_pop = ea.step(new_pop, i+P.backprop_epochs)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_pop_digits = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
elif P.select_opt == 'loss':
    best_pop_digits = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)

# Save model and results dict
ea_reservoir_model = open('models/digits_EA_reservoir_model.pkl', 'wb')
pickle.dump(best_pop_digits, ea_reservoir_model)
ea_reservoir_model.close()

# -----------------------------------------------------------------------------------------------------------
# Run baseline model
# -----------------------------------------------------------------------------------------------------------

bl_model_digits = Net.Baseline_RNN(P.input_size_digits, P.n_hidden, P.n_labels, P.T, dataset = 'Digits')
optimizer_digits = optim.SGD([p for p in bl_model_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD, momentum=P.momentum_SGD)
trained_bl_digits = Ops.training(bl_model_digits, train_loader_digits, val_loader_digits, P.n_epochs, optimizer_digits, P.loss_function, P.max_loss_iter)

# Save model and results dict
baseline_model = open('models/digits_baseline_model.pkl', 'wb')
pickle.dump(trained_bl_digits, baseline_model)
baseline_model.close()

# -----------------------------------------------------------------------------------------------------------
# Run RNN without evo
# -----------------------------------------------------------------------------------------------------------

res_model_digits = Net.Reservoir_RNN(P.input_size_digits, P.reservoir_size, P.n_labels, P.T, dataset = 'Digits')
optimizer_digits = optim.SGD([p for p in res_model_digits.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                             momentum=P.momentum_SGD)
trained_res_digits = Ops.training(res_model_digits, train_loader_digits, val_loader_digits, P.n_epochs,
                                  optimizer_digits, P.loss_function, P.max_loss_iter)

# Save model and results dict
reservoir_model_no_evo = open('models/digits_reservoir_model_no_evo.pkl', 'wb')
pickle.dump(trained_res_digits, reservoir_model_no_evo)
reservoir_model_no_evo.close()

# ----------------------------------------------------------------------------------------------------------
# Plot the results and save other data to results text file
# -----------------------------------------------------------------------------------------------------------

baseline_model_file = open('models/digits_baseline_model.pkl', 'rb')
baseline_model = pickle.load(baseline_model_file)
reservoir_model_no_evo_file = open('models/digits_reservoir_model_no_evo.pkl', 'rb')
reservoir_model_no_evo = pickle.load(reservoir_model_no_evo_file)
ea_reservoir_model_file = open('models/digits_EA_reservoir_model.pkl', 'rb')
ea_reservoir_model = pickle.load(ea_reservoir_model_file)

Ops.combined_plot_exp3(baseline_model['epoch'], baseline_model['loss_results'], reservoir_model_no_evo['loss_results'],
                         ea_reservoir_model, border=None, title='Digits pop %s - epoch %s - mutateopt %s - selectopt %s'
                                                                %(P.population_size, P.n_epochs,
                                                                  P.mutate_opt, P.select_mech))

# Save the test results + parameter settings to a file:
sys.stdout = open("plots/exp3/results/digits_ep_%s_pop_%s_mutateopt_%s_selectopt_%s.txt" %(P.n_epochs, P.population_size, P.mutate_opt, P.select_mech), "w")
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
      'mu: %s\n'
      'sigma: %s\n'
      'select opt: %s\n'
      'select mech: %s\n'
      'k_best: %s\n'
      'offspring ratio: %s\n'
      'n epochs: %s\n' % (P.population_size,P.generations , P.mutate_opt, P.perturb_rate,
P.mutate_bias, P.sample_dist, P.mu, P.sigma,  P.select_opt,  P.select_mech,  P.k_best,  P.offspring_ratio, P.n_epochs))

test_result_digits2 = Ops.evaluation(test_loader_digits, baseline_model['model'], 'Final score Digits on test set- baseline', P.loss_function)
test_result_digits = Ops.evaluation(test_loader_digits, reservoir_model_no_evo['model'], 'Final score Digits on test set - only output train', P.loss_function)
test_result_digits3 = Ops.evaluation(test_loader_digits, ea_reservoir_model[0]['model'], 'Final score Digits on test set- with evolution', P.loss_function)

# Baseline RNN model
print(summary(baseline_model['model'], torch.zeros(1, 64), show_input=True, show_hierarchical=False))
# Reservoir RNN model
print(summary(reservoir_model_no_evo['model'], torch.zeros(1, 64), show_input=True, show_hierarchical=False))

sys.stdout.close()

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

print('\nExecution time was: (hours:minute:seconds:microseconds) %s \n' %exc_time)

