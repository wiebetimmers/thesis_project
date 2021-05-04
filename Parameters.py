import torch.optim as optim
import torch.nn as nn
import Datasets as Data


# Data parameters
input_size_digits = 64  # 8 * 8 pixels
input_size_mnist = 784  # 28 * 28 pixels
n_labels = 10
batch_size= 50
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(batch_size)
LABELS = Data.get_labels()

# Network / Learning parameters
reservoir_size = 128 # for reservoir
n_hidden = 128 # for baseline
lr_SGD = 0.0001
momentum_SGD = 0.9
backprop_epochs = 5 # with backprop, only applicable in the evolutionairy approach.
T = 5             # amount of recurrent layers
loss_function = nn.NLLLoss(reduction='sum')
max_loss_iter = 10  # not really used yet


# EA parameters
population_size = 20
generations = 95        # epochs without backprop

mutate_opt = 'diff_mutation'  # Options: 'random_perturbation' , 'diff_mutation'
perturb_rate = 0.5 #If using diff mutation, use between 0-2
sample_dist = 'uniform'       # If using random perturbation, options: 'gaussian' , 'uniform'
select_opt = 'classification_error'  # 'classification_error' or 'loss'
select_mech = 'keep_k_best'   # Options: 'keep_k_best'= half of the parents , 'merge_all'
k_best = population_size // 4   # Keep 1/4 of the population as the best for new pop, the rest is new.
offspring_ratio = 2    # When mutating, create k times as much children as there are parents.


# Run all models for same amount of time, but make a distinction in epochs for the evolutionairy approach .
n_epochs = backprop_epochs + generations