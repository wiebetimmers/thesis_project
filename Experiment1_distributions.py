import datetime  # Keep track of execution time.
import sys
import Operations as Ops
import Datasets as Data
import Parameters as P
import pickle
from EA import EA

begin_time = datetime.datetime.now()

# Load data
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)

# --------------------------------------------------------------------------------------------------------------

ea = EA(P.population_size, val_loader_digits, P.loss_function, P.input_size_digits, P.reservoir_size, P.n_labels)

# Running an experiment for the distributions!
distributions = ['gaussian', 'uniform']

for dist in distributions:
    # Start with the same backpropped population for each distribution
    reservoir_model = open('models/digits_reservoir_model_RP_start.pkl', 'rb')
    reservoir_pop = pickle.load(reservoir_model)

    print('\nStart training the %s random perturb model.\n' % dist)

    # Check to set up parameters correctly
    P.mutate_opt = 'random_perturbation'
    P.sample_dist = dist

    # Initialize the population - same initialization for each dist
    new_pop = reservoir_pop

    # Perform ea steps
    for i in range(P.generations):
        new_pop = ea.step(new_pop, i+P.backprop_epochs)

    # Sort population after x amount of generations, based on classification error or loss performance
    if P.select_opt == 'classification_error':
        best_pop_digits = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
    else:  # Log loss
        best_pop_digits = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)

    # Save model and results dict
    ea_reservoir_model = open('models/exp1/digits_EA_reservoir_model_RP_%s_s%s_pr%s.pkl' %
                              (dist, P.sigma, P.perturb_rate_decay), 'wb')
    pickle.dump(best_pop_digits, ea_reservoir_model)
    ea_reservoir_model.close()

# -----------------------------------------------------------------------------------------------------------

# Plot the results

gaussian_file = open('models/exp1/digits_EA_reservoir_model_RP_gaussian_s%s_pr%s.pkl'
                     % (P.sigma, P.perturb_rate_decay), 'rb')
gaussian_model = pickle.load(gaussian_file)
uniform_file = open('models/exp1/digits_EA_reservoir_model_RP_uniform_s%s_pr%s.pkl'
                    % (P.sigma, P.perturb_rate_decay), 'rb')
uniform_model = pickle.load(uniform_file)

sys.stdout = open('plots/exp1/results/experiment distributions-Digits pop %s-epoch %s-mutateopt %s '
                  '-select %s-decay %s-sigma %s'
                  % (P.population_size, P.n_epochs, P.mutate_opt, P.select_mech, P.perturb_rate_decay, P.sigma), "w")

Ops.print_parameters()

Ops.plot_loss_exp1(gaussian_model, uniform_model, title='Digits pop %s - epoch %s - '
                                                        'mutateopt %s - selectopt %s - decay %s - sigma %s'
                                                        % (P.population_size, P.n_epochs, P.mutate_opt, P.select_mech,
                                                            P.perturb_rate_decay, P.sigma))

# Add additional printing of lowest loss value we got
print('Final score Digits on test set- gaussian')
test_result_digits = Ops.evaluation(test_loader_digits, gaussian_model[0]['model'],
                                    '\n\nFinal score Digits on test set- gaussian', P.loss_function, test_set=True)
print('\nFinal score Digits on test set- uniform')
test_result_digits1 = Ops.evaluation(test_loader_digits, uniform_model[0]['model'],
                                     '\n\nFinal score Digits on test set- uniform', P.loss_function, test_set=True)

# ----------------------------------------------------------------------------------------------------------

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

print('\nExecution time was: (hours:minute:seconds:microseconds) %s \n' % exc_time)

sys.stdout.close()
