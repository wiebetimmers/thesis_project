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


# -----------------------------------------------------------------------------------------------------------
# Run baseline model
# -----------------------------------------------------------------------------------------------------------

baseline_set_digits = []

for i in range(P.population_size):
    bl_model_digits = Net.Baseline_RNN(P.input_size_digits, P.n_hidden, P.n_labels, P.T, dataset='Digits')
    optimizer_digits = optim.SGD([p for p in bl_model_digits.parameters() if p.requires_grad],
                                 lr=P.lr_SGD, momentum=P.momentum_SGD)
    trained_bl_digits = Ops.training(bl_model_digits, train_loader_digits, val_loader_digits, P.n_epochs,
                                     optimizer_digits, P.loss_function, P.max_loss_iter)
    baseline_set_digits.append(trained_bl_digits)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_baseline_digits = sorted(baseline_set_digits, key=lambda k: k['class_error_results'][-1], reverse=False)
else:  # Log loss
    best_baseline_digits = sorted(baseline_set_digits, key=lambda k: k['loss_results'][-1], reverse=False)

# Save model and results dict
baseline_model = open('models/exp3/digits_baseline_model.pkl', 'wb')
pickle.dump(best_baseline_digits, baseline_model)
baseline_model.close()
