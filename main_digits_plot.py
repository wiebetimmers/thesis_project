import datetime  # Keep track of execution time.
import sys
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

baseline_model_file = open('models/baseline_model.pkl', 'rb')
baseline_model = pickle.load(baseline_model_file)
reservoir_model_no_evo_file = open('models/reservoir_model_no_evo.pkl', 'rb')
reservoir_model_no_evo = pickle.load(reservoir_model_no_evo_file)
ea_reservoir_model_file = open('models/EA_reservoir_model.pkl', 'rb')
ea_reservoir_model = pickle.load(ea_reservoir_model_file)

# Plot above plots in one plot
Ops.combined_plot_result(
            baseline_model['epoch'],
            baseline_model['loss_results'],
            baseline_model['class_error_results'],
            reservoir_model_no_evo['loss_results'],
            reservoir_model_no_evo['class_error_results'],
            ea_reservoir_model[0]['loss_results'],
            ea_reservoir_model[0]['class_error_results'],
            border = P.backprop_epochs,
            label_bl = 'Baseline RNN',
            label_res = 'Reservoir RNN',
            label_evo = 'EA Reservoir RNN',
            title = 'Digits pop %s - epoch %s - mutateopt %s - selectopt %s' %(P.population_size, P.n_epochs, P.mutate_opt, P.select_mech))

Ops.best_pop_plot(ea_reservoir_model,
              ea_reservoir_model[0],
              title='Complete Digits pop %s - epoch %s-  mutate opt %s - selectopt %s' %(P.population_size, P.n_epochs, P.mutate_opt, P.select_mech))

# Save the test results + parameter settings to a file:
sys.stdout = open("plots/results_digits_ep_%s_pop_%s.txt" %(P.n_epochs, P.population_size), "w")
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