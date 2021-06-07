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

# Open initialized 5 epochs
initial_model = open('models/exp3/digits_initial_model_start_pop_%s.pkl' % P.population_size, 'rb')
initial_models = pickle.load(initial_model)

# -----------------------------------------------------------------------------------------------------------
# Run RNN without evo
# -----------------------------------------------------------------------------------------------------------

all_epochs_models = []

for i in initial_models:
    optimizer_digits = optim.SGD([p for p in i['model'].parameters() if p.requires_grad], lr=P.lr_SGD,
                             momentum=P.momentum_SGD)
    trained_res_digits = Ops.training(i['model'], train_loader_digits, val_loader_digits, P.n_epochs,
                                  optimizer_digits, P.loss_function, P.max_loss_iter, exp3=True, initial_model=i)
    all_epochs_models.append(trained_res_digits)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_res_digits = sorted(all_epochs_models, key=lambda k: k['class_error_results'][-1], reverse=False)
else:  # Log loss
    best_res_digits = sorted(all_epochs_models, key=lambda k: k['loss_results'][-1], reverse=False)

# Save model and results dict
reservoir_model_no_evo = open('models/exp3/digits_reservoir_model_no_evo.pkl', 'wb')
pickle.dump(best_res_digits, reservoir_model_no_evo)
reservoir_model_no_evo.close()

