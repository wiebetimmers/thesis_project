import datetime  # Keep track of execution time.
begin_time = datetime.datetime.now()
import sys
import torch.optim as optim
import Networks as Net
import Operations as Ops
import Datasets as Data
import Parameters as P
import pickle
from EA import EA

# Load data
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)

# -----------------------------------------------------------------------------------------------------

# Running an experiment for the distributions!
distributions = ['gaussian', 'uniform', 'cauchy', 'lognormal']

for dist in distributions:

    print('\nStart training the %s random perturb model.\n' %dist)
    # Check to set up paramters correctly
    P.mutate_opt = 'random_perturbation'
    P.sample_dist = dist

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
    ea_reservoir_model = open('models/digits_EA_reservoir_model_RP_%s.pkl' %dist, 'wb')
    pickle.dump(best_pop_digits, ea_reservoir_model)
    ea_reservoir_model.close()

# -----------------------------------------------------------------------------------------------------------

# Plot the results
distributions = ['gaussian', 'uniform', 'cauchy', 'lognormal']

gaussian_file = open('models/digits_EA_reservoir_model_RP_gaussian.pkl', 'rb')
gaussian_model = pickle.load(gaussian_file)
uniform_file = open('models/digits_EA_reservoir_model_RP_uniform.pkl', 'rb')
uniform_model = pickle.load(uniform_file)
cauchy_file = open('models/digits_EA_reservoir_model_RP_cauchy.pkl', 'rb')
cauchy_model = pickle.load(cauchy_file)
lognormal_file = open('models/digits_EA_reservoir_model_RP_lognormal.pkl', 'rb')
lognormal_model = pickle.load(lognormal_file)

# Plot above models in one plot
Ops.combined_plot_result_distributions(gaussian_model[0]['epoch'],
            gaussian_model[0]['loss_results'],
            gaussian_model[0]['class_error_results'],
            uniform_model[0]['loss_results'],
            uniform_model[0]['class_error_results'],
            cauchy_model[0]['loss_results'],
            cauchy_model[0]['class_error_results'],
            lognormal_model[0]['loss_results'],
            lognormal_model[0]['class_error_results'],
            border = P.backprop_epochs,
            label_gaussian = 'Gaussian',
            label_uniform = 'Uniform',
            label_cauchy = 'Cauchy',
            label_lognormal= 'Lognormal',
                                   title = 'Experiment distributions - Digits pop %s - epoch %s - '
                                           'mutateopt %s - selectopt %s - decay %s - sigma %s'
                                           %(P.population_size, P.n_epochs, P.mutate_opt, P.select_mech, P.perturb_rate_decay, P.sigma))


sys.stdout = open('plots/experiment distributions-Digits pop %s-epoch %s-mutateopt %s -select %s-decay %s-sigma %s'
                                           %(P.population_size, P.n_epochs, P.mutate_opt, P.select_mech, P.perturb_rate_decay, P.sigma), "w")

# Add additional printing of lowest loss value we got

test_result_digits = Ops.evaluation(test_loader_digits, gaussian_model[0]['model'], '\n\nFinal score Digits on test set- gaussian', P.loss_function)
test_result_digits1 = Ops.evaluation(test_loader_digits, uniform_model[0]['model'], 'Final score Digits on test set- uniform', P.loss_function)
test_result_digits2 = Ops.evaluation(test_loader_digits, cauchy_model[0]['model'], 'Final score Digits on test set- cauchy', P.loss_function)
test_result_digits3 = Ops.evaluation(test_loader_digits, lognormal_model[0]['model'], 'Final score Digits on test set- lognormal', P.loss_function)

# ----------------------------------------------------------------------------------------------------------

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

print('\nExecution time was: (hours:minute:seconds:microseconds) %s \n' %exc_time)

sys.stdout.close()