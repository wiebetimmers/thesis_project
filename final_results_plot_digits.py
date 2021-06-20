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
import matplotlib.pyplot as plt
from EA import EA

begin_time = datetime.datetime.now()

# Load data
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)

baseline_model_file = open('models/exp3/digits_baseline_model.pkl', 'rb')
baseline_model = pickle.load(baseline_model_file)
reservoir_model_no_evo_file = open('models/exp3/digits_reservoir_model_no_evo.pkl', 'rb')
reservoir_model_no_evo = pickle.load(reservoir_model_no_evo_file)
ea_reservoir_model_file = open('models/exp3/digits_EA_reservoir_model_offspring%s_biasmutate%s.pkl' %
                               (P.offspring_ratio, P.mutate_bias),
                               'rb')
ea_reservoir_model = pickle.load(ea_reservoir_model_file)


# Save the test results + parameter settings to a file:

sys.stdout = open("plots/exp3/results/digits_ep_%s_pop_%s_mutatebias_%s_offspring_%s.txt" % (P.n_epochs,
                                                                                             P.population_size,
                                                                                             P.mutate_bias,
                                                                                             P.offspring_ratio), "w")

Ops.print_parameters()

test_result_digits2 = Ops.evaluation(test_loader_digits, baseline_model[0]['model'],
                                     'Final score Digits on test set- baseline', P.loss_function)
test_result_digits = Ops.evaluation(test_loader_digits, reservoir_model_no_evo[0]['model'],
                                    'Final score Digits on test set - only output train', P.loss_function)
test_result_digits3 = Ops.evaluation(test_loader_digits, ea_reservoir_model[0]['model'],
                                     'Final score Digits on test set- with evolution', P.loss_function)

# and also the 5 epoch thing, have we learned something at all?
# Open initialized 5 epochs
initial_model = open('models/exp3/digits_initial_model_start_pop_%s.pkl' % P.population_size, 'rb')
initial_models = pickle.load(initial_model)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_initial = sorted(initial_models, key=lambda k: k['class_error_results'][-1], reverse=False)
else:  # Log loss
    best_initial = sorted(initial_models, key=lambda k: k['loss_results'][-1], reverse=False)

# Print population stats and plot results
Ops.print_stat_initial(best_initial)

test_result_digits4 = Ops.evaluation(test_loader_digits, best_initial[0]['model'],
                                     'Final score Digits on test set- the intial 5 epoch model', P.loss_function)

Ops.plot_loss_final(baseline_model, reservoir_model_no_evo, ea_reservoir_model,
                    title='offspring_ratio_%s_mutatebias_%s' % (P.offspring_ratio, P.mutate_bias))

# Baseline RNN model
print(summary(baseline_model[0]['model'], torch.zeros(1, 64), show_input=False, show_hierarchical=False))
# Reservoir RNN model
print(summary(reservoir_model_no_evo[0]['model'], torch.zeros(1, 64), show_input=False, show_hierarchical=False))

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

print('\nExecution time was: (hours:minute:seconds:microseconds) %s \n' % exc_time)

sys.stdout.close()
