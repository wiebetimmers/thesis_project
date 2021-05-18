import torch.optim as optim
import torch.nn as nn
import Datasets as Data

# Data parameters
input_size_digits = 64  # 8 * 8 pixels
input_size_mnist = 784  # 28 * 28 pixels
n_labels = 10
batch_size= 50
LABELS = Data.get_labels()

# Network / Learning parameters
reservoir_size = 128 # for reservoir
n_hidden = 128 # for baseline
lr_SGD = 0.0001
momentum_SGD = 0.9
backprop_epochs = 5 # with backprop, only applicable in the evolutionairy approach.
T = 5             # amount of recurrent layers
loss_function = nn.NLLLoss(reduction='sum')

max_loss_iter = 10  # not  used yet


# EA parameters
population_size = 10
generations = 95     # epochs without backprop
mutate_opt = 'random_perturbation'  # Options: 'random_perturbation' , 'diff_mutation'
perturb_rate = 0.5  # initial rate, Fraction of sample mutation added to the population
perturb_rate_decay = 0.1
mutate_bias = False  # If we want to mutate/crossover the bias or not.
sample_dist = 'gaussian'       # If using random perturbation, options: 'gaussian', 'uniform', 'cauchy', 'lognormal'
mu = 0  # If we require a mu
sigma = 0.05  # If we require a variance
select_opt = 'loss'  # 'classification_error' or 'loss'
select_mech = 'keep_k_best'   # Options: 'keep_k_best'= k best of the parents , 'merge_all'
k_best = population_size // 5   # Keep 1/k of the population as the best for new pop, the rest is new.
offspring_ratio = 1    # Optional, increase the offspring by factor k (k times as much offspring)


# Run all models for same amount of time, but make a distinction in epochs for the evolutionairy approach .
n_epochs = backprop_epochs + generations
