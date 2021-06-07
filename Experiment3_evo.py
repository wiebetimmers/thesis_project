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
from EA import EA

begin_time = datetime.datetime.now()

# Load data
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)
ea = EA(P.population_size, val_loader_digits, P.loss_function, P.input_size_digits, P.reservoir_size, P.n_labels)

# Open initialized 5 epochs
initial_model = open('models/exp3/digits_initial_model_start_pop_%s.pkl' % P.population_size, 'rb')
new_pop = pickle.load(initial_model)

# Perform ea steps
for i in range(P.generations):
    new_pop = ea.step(new_pop, i+P.backprop_epochs)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_pop_digits = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
else:  # Log loss
    best_pop_digits = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)

# Save model and results dict
ea_reservoir_model = open('models/exp3/digits_EA_reservoir_model_offspring%s_biasmutate%s.pkl' %(P.offspring_ratio,
                                                                              P.mutate_bias), 'wb')
pickle.dump(best_pop_digits, ea_reservoir_model)
ea_reservoir_model.close()

