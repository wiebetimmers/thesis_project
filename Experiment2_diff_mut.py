import datetime  # Keep track of execution time.
begin_time = datetime.datetime.now()
import sys
import Operations as Ops
import Datasets as Data
import Parameters as P
import pickle
from EA import EA

# Load data
train_loader_digits, val_loader_digits, test_loader_digits = Data.get_digits_loaders(P.batch_size)

# --------------------------------------------------------------------------------------------------------------

ea = EA(P.population_size, val_loader_digits, P.loss_function, P.input_size_digits, P.reservoir_size, P.n_labels)

# Running an experiment for the mutations!
mutations = ['random_perturbation', 'diff_mutation']

for mutation in mutations:
    # Start with the same backpropped population for each distribution
    reservoir_model = open('models/digits_reservoir_model_RP_start.pkl', 'rb')
    reservoir_pop = pickle.load(reservoir_model)

    print('\nStart training the %s model.\n' %mutation)

    # Check to set up paramters correctly
    P.mutate_opt = mutation

    # Initialize the population - same intialization for each dist
    new_pop = reservoir_pop

    # Perform ea steps
    for i in range(P.generations):
        new_pop = ea.step(new_pop, i+P.backprop_epochs)

    # Sort population after x amount of generations, based on classification error or loss performance
    if P.select_opt == 'classification_error':
        best_pop_digits = sorted(new_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
    elif P.select_opt == 'loss':
        best_pop_digits = sorted(new_pop, key=lambda k: k['loss_results'][-1], reverse=False)

    # Save model and results dict
    ea_reservoir_model = open('models/exp2/digits_EA_reservoir_model_%s.pkl' %mutation, 'wb')
    pickle.dump(best_pop_digits, ea_reservoir_model)
    ea_reservoir_model.close()

# ----------------------------------------------------------------------------------------------------------

# Plot the results

random_pert_file = open('models/exp2/digits_EA_reservoir_model_random_perturbation.pkl', 'rb')
random_pert_model = pickle.load(random_pert_file)
diff_mut_file = open('models/exp2/digits_EA_reservoir_model_diff_mutation.pkl', 'rb')
diff_mut_model = pickle.load(diff_mut_file)

sys.stdout = open('plots/exp2/results/experiment mutations-Digits pop %s-epoch - %s -select %s-decay %s-sigma %s'
                                           %(P.population_size, P.n_epochs,  P.select_mech, P.perturb_rate_decay, P.sigma), "w")

Ops.print_parameters()

Ops.plot_loss_exp2(random_pert_model, diff_mut_model, title= 'Digits pop %s - epoch %s - '
                                           'mutateopt %s - selectopt %s - decay %s - sigma %s'
                                           %(P.population_size, P.n_epochs, P.mutate_opt, P.select_mech, P.perturb_rate_decay, P.sigma))

# Add additional printing of lowest loss value we got

print('Final score Digits on test set- random perturbation')
test_result_digits = Ops.evaluation(test_loader_digits, random_pert_model[0]['model'],
                                    '\n\nFinal score Digits on test set- random pert', P.loss_function, test_set=True)
print('Final score Digits on test set- diff mutation')
test_result_digits1 = Ops.evaluation(test_loader_digits, diff_mut_model[0]['model'],
                                     '\n\nFinal score Digits on test set- diff mutation', P.loss_function,test_set=True)

# ----------------------------------------------------------------------------------------------------------

# Print execution time:
exc_time = datetime.datetime.now() - begin_time

print('\nExecution time was: (hours:minute:seconds:microseconds) %s \n' %exc_time)

sys.stdout.close()