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

ea_reservoir_model_file1 = open('models/exp3/digits_EA_reservoir_model_offspring1_biasmutate%s.pkl' % (
                                                                              P.mutate_bias), 'rb')
ea_reservoir_model1 = pickle.load(ea_reservoir_model_file1)
ea_reservoir_model_file2 = open('models/exp3/digits_EA_reservoir_model_offspring2_biasmutate%s.pkl' % (
                                                                              P.mutate_bias), 'rb')
ea_reservoir_model2 = pickle.load(ea_reservoir_model_file2)
ea_reservoir_model_file3 = open('models/exp3/digits_EA_reservoir_model_offspring3_biasmutate%s.pkl' % (
                                                                              P.mutate_bias), 'rb')
ea_reservoir_model3 = pickle.load(ea_reservoir_model_file3)


# Save the test results + parameter settings to a file:

sys.stdout = open("plots/exp3/results/digits_ep_%s_pop_%s_mutatebias_%s_offspring_all.txt" %
                  (P.n_epochs, P.population_size, P.mutate_bias), "w")
# Plot and print

Ops.print_parameters()

test_result_digits1 = Ops.evaluation(test_loader_digits, ea_reservoir_model1[0]['model'],
                                     'Final score Digits on test set- with evolution - offspring 1', P.loss_function)
test_result_digits2 = Ops.evaluation(test_loader_digits, ea_reservoir_model2[0]['model'],
                                     'Final score Digits on test set- with evolution - offspring 2', P.loss_function)
test_result_digits3 = Ops.evaluation(test_loader_digits, ea_reservoir_model3[0]['model'],
                                     'Final score Digits on test set- with evolution - offspring 3', P.loss_function)

Ops.plot_loss_exp3(ea_reservoir_model1, ea_reservoir_model2, ea_reservoir_model3,
                   title='mutatebias_%s' % P.mutate_bias)


sys.stdout.close()
