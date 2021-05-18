import datetime  # Keep track of execution time.
begin_time = datetime.datetime.now()

import torch.optim as optim
import Networks as Net
import Operations as Ops
import Datasets as Data
import Parameters as P
import pickle
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
    new_pop = ea.step(new_pop, i+P.backprop_epochs)

# Sort population after x amount of generations, based on classification error or loss performance
if P.select_opt == 'classification_error':
    best_pop_mnist = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
elif P.select_opt == 'loss':
    best_pop_mnist = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)

# Save model and results dict
ea_reservoir_model = open('models/mnist_EA_reservoir_model.pkl', 'wb')
pickle.dump(best_pop_mnist, ea_reservoir_model)
ea_reservoir_model.close()

# -----------------------------------------------------------------------------------------------------------

# Run baseline model
bl_model_mnist = Net.Baseline_RNN(P.input_size_mnist, P.n_hidden, P.n_labels, P.T, dataset = 'MNIST')
optimizer_mnist = optim.SGD([p for p in bl_model_mnist.parameters() if p.requires_grad == True], lr=P.lr_SGD, momentum=P.momentum_SGD)
trained_bl_mnist = Ops.training(bl_model_mnist, train_loader_mnist, val_loader_mnist, P.n_epochs, optimizer_mnist, P.loss_function, P.max_loss_iter)

# Save model and results dict
baseline_model = open('models/mnist_baseline_model.pkl', 'wb')
pickle.dump(trained_bl_mnist, baseline_model)
baseline_model.close()

# -----------------------------------------------------------------------------------------------------------

# Run RNN without evo
res_model_mnist= Net.Reservoir_RNN(P.input_size_mnist, P.reservoir_size, P.n_labels, P.T, dataset = 'MNIST')
optimizer_mnist = optim.SGD([p for p in res_model_mnist.parameters() if p.requires_grad == True], lr=P.lr_SGD,
                             momentum=P.momentum_SGD)
trained_res_mnist = Ops.training(res_model_mnist, train_loader_mnist, val_loader_mnist, P.n_epochs,
                                  optimizer_mnist, P.loss_function, P.max_loss_iter)

# Save model and results dict
reservoir_model_no_evo = open('models/mnist_reservoir_model_no_evo.pkl', 'wb')
pickle.dump(trained_res_mnist, reservoir_model_no_evo)
reservoir_model_no_evo.close()

# ----------------------------------------------------------------------------------------------------------

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

print('Execution time was: (hours:minute:seconds:microseconds) %s ' %exc_time)
