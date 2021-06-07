import matplotlib.pyplot as plt
import torch
import numpy as np
import Datasets as Data
import statistics as st
import Parameters as P

LABELS = Data.get_labels()
plt.rcParams["figure.figsize"] = (5, 3)


# Function that transforms the tensor output to a predicted target name.
def category_from_output(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return LABELS[category_i]


def get_stats(pop, initial=False):
    min_list = []
    max_list = []
    mean_list = []
    std_list = []
    if not initial:
        epochs = P.n_epochs
    else:
        epochs = P.backprop_epochs

    for i in range(epochs):
        epoch_list = []
        for j in range(len(pop)):
            epoch_list.append(pop[j]['class_error_results'][i])
        min_list.append(min(epoch_list))
        max_list.append(max(epoch_list))
        mean_list.append(st.mean(epoch_list))
        std_list.append(st.stdev(epoch_list))

    stats_results = {
        'epoch': np.array(range(P.n_epochs)),
        'min_list': np.array(min_list),
        'max_list': np.array(max_list),
        'mean_list': np.array(mean_list),
        'std_list': np.array(std_list)}

    return stats_results


def plot_loss_exp1(gaussian, uniform, title=''):
    stats_gaussian = get_stats(gaussian)
    stats_uniform = get_stats(uniform)

    plt.plot(stats_gaussian['epoch'], stats_gaussian['mean_list'], 'b-', label='Gaussian')
    plt.fill_between(stats_gaussian['epoch'], stats_gaussian['min_list'],
                     stats_gaussian['max_list'], color='b', alpha=0.2)

    best_gaussian = gaussian[0]['loss_results'][-1]
    worst_gaussian = gaussian[-1]['loss_results'][-1]
    print('Best val loss gaussian: %s' % best_gaussian)
    print('Worst val loss gaussian: %s' % worst_gaussian)
    print('Mean Last population gaussian: %s, std: %s' % (stats_gaussian['mean_list'][-1],
                                                          stats_gaussian['std_list'][-1]))

    plt.plot(stats_uniform['epoch'], stats_uniform['mean_list'], 'r-', label='Uniform')
    plt.fill_between(stats_uniform['epoch'], stats_uniform['min_list'],
                     stats_uniform['max_list'], color='r', alpha=0.2)

    best_uniform = uniform[0]['loss_results'][-1]
    worst_uniform = uniform[-1]['loss_results'][-1]
    print('Best val loss uniform: %s' % best_uniform)
    print('Worst val loss uniform: %s' % worst_uniform)
    print('Mean Last population uniform: %s, std: %s\n\n' % (stats_uniform['mean_list'][-1],
                                                             stats_uniform['std_list'][-1]))

    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.legend(loc='upper right')
    plt.title(r'$\alpha = %s , \sigma$ = %s' % (P.perturb_rate_decay, P.sigma))
    plt.savefig('plots/exp1/%s.png' % title, bbox_inches='tight')

    return best_gaussian, stats_gaussian, best_uniform, stats_uniform


def plot_loss_exp2(random_pert, diff_mut, title=''):
    stats_random_pert = get_stats(random_pert)
    stats_diff_mut = get_stats(diff_mut)

    plt.plot(stats_random_pert['epoch'], stats_random_pert['mean_list'], 'g-', label='Random Perturbation')
    plt.fill_between(stats_random_pert['epoch'], stats_random_pert['min_list'],
                     stats_random_pert['max_list'], color='b', alpha=0.2)

    best_random_pert = random_pert[0]['loss_results'][-1]
    worst_random_pert = random_pert[-1]['loss_results'][-1]
    print('Best val loss random perturbation: %s' % best_random_pert)
    print('Worst val loss random perturbation: %s' % worst_random_pert)
    print('Mean last population random perturbation: %s, std: %s' % (stats_random_pert['mean_list'][-1],
                                                                     stats_random_pert['std_list'][-1]))

    plt.plot(stats_diff_mut['epoch'], stats_diff_mut['mean_list'], 'm-', label='Differential Mutation')
    plt.fill_between(stats_diff_mut['epoch'], stats_diff_mut['min_list'],
                     stats_diff_mut['max_list'], color='r',
                     alpha=0.2)

    best_diff_mut = diff_mut[0]['loss_results'][-1]
    worst_diff_mut = diff_mut[-1]['loss_results'][-1]
    print('Best val loss diff mutation: %s' % best_diff_mut)
    print('Worst val loss diff mutation: %s' % worst_diff_mut)
    print('Mean Last population diff mutation: %s, std: %s\n\n' % (stats_diff_mut['mean_list'][-1],
                                                                   stats_diff_mut['std_list'][-1]))

    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.legend(loc='upper right')
    plt.title(r'$\alpha = %s$, %s selection' % (P.perturb_rate_decay, P.select_mech))
    plt.savefig('plots/exp2/%s.png' % title, bbox_inches='tight')
    return


def plot_loss_exp3(ea_reservoir1,
                   ea_reservoir2,
                   ea_reservoir3,
                   title=''):
    stats_ea_reservoir1 = get_stats(ea_reservoir1)
    stats_ea_reservoir2 = get_stats(ea_reservoir2)
    stats_ea_reservoir3 = get_stats(ea_reservoir3)

    plt.plot(stats_ea_reservoir1['epoch'], stats_ea_reservoir1['mean_list'], 'b-', label=r'$\psi$ = 1')
    plt.fill_between(stats_ea_reservoir1['epoch'], stats_ea_reservoir1['min_list'],
                     stats_ea_reservoir1['max_list'], color='b', alpha=0.2)

    best_ea_reservoir1 = ea_reservoir1[0]['loss_results'][-1]
    worst_ea_reservoir1 = ea_reservoir1[-1]['loss_results'][-1]
    print('\nBest val loss baseline: %s' % best_ea_reservoir1)
    print('Worst val loss baseline: %s' % worst_ea_reservoir1)
    print('Mean baseline: %s, std: %s\n' % (stats_ea_reservoir1['mean_list'][-1],
                                            stats_ea_reservoir1['std_list'][-1]))

    plt.plot(stats_ea_reservoir2['epoch'], stats_ea_reservoir2['mean_list'], 'r-', label=r'$\psi$ = 2')
    plt.fill_between(stats_ea_reservoir2['epoch'], stats_ea_reservoir2['min_list'],
                     stats_ea_reservoir2['max_list'], color='r',
                     alpha=0.2)

    best_ea_reservoir2 = ea_reservoir2[0]['loss_results'][-1]
    worst_ea_reservoir2 = ea_reservoir2[-1]['loss_results'][-1]
    print('\nBest val loss reservoir RNN: %s' % best_ea_reservoir2)
    print('Worst val loss reservoir RNN: %s' % worst_ea_reservoir2)
    print('Mean Last population reservoir RNN: %s, std: %s\n\n' % (stats_ea_reservoir2['mean_list'][-1],
                                                                   stats_ea_reservoir2['std_list'][-1]))

    plt.plot(stats_ea_reservoir3['epoch'], stats_ea_reservoir3['mean_list'], 'g-', label=r'$\psi$ = 3')
    plt.fill_between(stats_ea_reservoir3['epoch'], stats_ea_reservoir3['min_list'],
                     stats_ea_reservoir3['max_list'], color='g',
                     alpha=0.2)

    best_ea_reservoir3 = ea_reservoir3[0]['loss_results'][-1]
    worst_ea_reservoir3 = ea_reservoir3[-1]['loss_results'][-1]
    print('\nBest val loss EA reservoir RNN: %s' % best_ea_reservoir3)
    print('Worst val loss EA reservoir RNN: %s' % worst_ea_reservoir3)
    print('Mean Last population reservoir RNN: %s, std: %s\n\n' % (stats_ea_reservoir3['mean_list'][-1],
                                                                   stats_ea_reservoir3['std_list'][-1]))

    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.ylim(0, 0.5)
    plt.legend(loc='upper right')
    plt.title('Mutate bias = %s' % P.mutate_bias)
    plt.savefig('plots/exp3/%s.png' % title, bbox_inches='tight')

    return


def plot_loss_final(baseline, reservoir, ea_reservoir, title=''):
    stats_baseline = get_stats(baseline)
    stats_reservoir = get_stats(reservoir)
    stats_ea_reservoir = get_stats(ea_reservoir)

    metric = 'loss_results'

    plt.plot(stats_baseline['epoch'], stats_baseline['mean_list'], 'b-', label='Baseline')
    plt.fill_between(stats_baseline['epoch'], stats_baseline['min_list'],
                     stats_baseline['max_list'], color='b', alpha=0.2)

    best_baseline = baseline[0][metric][-1]
    worst_baseline = baseline[-1][metric][-1]
    print('\nBest val loss baseline: %s' % best_baseline)
    print('Worst val loss baseline: %s' % worst_baseline)
    print('Mean baseline: %s, std: %s\n' % (stats_baseline['mean_list'][-1], stats_baseline['std_list'][-1]))

    plt.plot(stats_reservoir['epoch'], stats_reservoir['mean_list'], 'r-', label='Reservoir')
    plt.fill_between(stats_reservoir['epoch'], stats_reservoir['min_list'],
                     stats_reservoir['max_list'], color='r',
                     alpha=0.2)

    best_reservoir = reservoir[0][metric][-1]
    worst_reservoir = reservoir[-1][metric][-1]
    print('\nBest val loss reservoir RNN: %s' % best_reservoir)
    print('Worst val loss reservoir RNN: %s' % worst_reservoir)
    print('Mean Last population reservoir RNN: %s, std: %s\n\n' % (stats_reservoir['mean_list'][-1],
                                                                   stats_reservoir['std_list'][-1]))

    plt.plot(stats_ea_reservoir['epoch'], stats_ea_reservoir['mean_list'], 'g-', label='EA Reservoir')
    plt.fill_between(stats_ea_reservoir['epoch'], stats_ea_reservoir['min_list'],
                     stats_ea_reservoir['max_list'], color='g',
                     alpha=0.2)

    best_ea_reservoir = ea_reservoir[0][metric][-1]
    worst_ea_reservoir = ea_reservoir[-1][metric][-1]
    print('\nBest val loss EA reservoir RNN: %s' % best_ea_reservoir)
    print('Worst val loss EA reservoir RNN: %s' % worst_ea_reservoir)
    print('Mean Last population reservoir RNN: %s, std: %s\n\n' % (stats_ea_reservoir['mean_list'][-1],
                                                                   stats_ea_reservoir['std_list'][-1]))

    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.ylim(0, 0.5)
    plt.legend(loc='upper right')
    plt.savefig('plots/exp3/%s.png' % title, bbox_inches='tight')

    return


def print_stat_initial(initial_model):
    stats = get_stats(initial_model, initial=True)
    best_initial = initial_model[0]['loss_results'][-1]
    worst_initial = initial_model[-1]['loss_results'][-1]
    print('\nBest val loss initial model: %s' % best_initial)
    print('Worst val loss initial model: %s' % worst_initial)
    print('Mean initial model: %s, std: %s\n\n' % (stats['mean_list'][-1], stats['std_list'][-1]))
    return


def print_parameters():
    print('Network parameters:\n'
          'reservoir size: %s, \n'
          'n_hidden: %s, \n'
          'learning rate: %s, \n'
          'momentum sgd: %s, \n'
          'backprop epochs: %s, \n'
          'T: %s, \n'
          'loss_function: %s \n' % (P.reservoir_size, P.n_hidden, P.lr_SGD, P.momentum_SGD,
                                    P.backprop_epochs, P.T, P.loss_function))
    print('EA parameters: \n'
          ' pop size: %s,\n'
          'generations: %s,\n'
          'mutate opt: %s,\n'
          'perturb rate: %s,\n'
          'perturb rate decay: %s, \n'
          'mutate_bias: %s\n'
          'sample_dist: %s\n'
          'mu: %s\n'
          'sigma: %s\n'
          'select opt: %s\n'
          'select mech: %s\n'
          'k_best: %s\n'
          'offspring ratio: %s\n'
          'n epochs: %s\n'
          % (P.population_size, P.generations, P.mutate_opt, P.perturb_rate, P.perturb_rate_decay,
             P.mutate_bias, P.sample_dist, P.mu, P.sigma, P.select_opt, P.select_mech, P.k_best,
             P.offspring_ratio, P.n_epochs))

    print('# --------------------------------------------------------------------------\n')
    return


# Concatenating the results of all batches in the lists, calculating the total accuracy.
def accuracy(pred_targets_list, gold_targets_list):
    total_correct = 0
    total_amount = 0

    zip_list = zip(pred_targets_list, gold_targets_list)
    for pred_targets, gold_targets in zip_list:
        total_correct += (pred_targets == gold_targets).float().sum()
        total_amount += len(pred_targets)

    total_accuracy = 100 * total_correct / total_amount

    return total_accuracy.item()


# Concatenating the results of all batches in the lists, calculating the classification error.
def class_error(pred_targets_list, gold_targets_list):
    total_error = 0
    total_amount = 0

    zip_list = zip(pred_targets_list, gold_targets_list)
    for pred_targets, gold_targets in zip_list:
        total_error += (pred_targets != gold_targets).float().sum()
        total_amount += len(pred_targets)

    total_class_error = (total_error / total_amount) * 100

    return total_class_error.item()


# Evaluation -> used for validation and test set.
def evaluation(val_loader, model, epoch, loss_function, test_set=False):
    # Evaluating our performance so far
    model.eval()

    # Store all results in a list to calculate the accuracy.
    pred_target_total_acc = []
    target_total_acc = []

    # Initialize counters / c
    loss = 0.
    n = 0.

    # Iterating over the validation set batches, acquiring tensor formatted results.
    for indx_batch, (batch, targets) in enumerate(val_loader):
        output = model.forward(batch)
        pred_targets = np.array([])
        for item in output:
            pred_targets = np.append(pred_targets, category_from_output(item))
        pred_targets = torch.from_numpy(pred_targets).int()

        # Calculating loss
        loss_t = loss_function(output, targets.long())
        loss = loss + loss_t.item()
        n = n + batch.shape[0]

        # Append the batch result to a list of all results
        pred_target_total_acc.append(pred_targets)
        target_total_acc.append(targets)

    # Store the loss corrected by its size
    loss = loss / n

    classification_error = class_error(pred_target_total_acc, target_total_acc)
    if not test_set:
        print('Epoch: %s - Loss of: %s - Classification Error of: %s' % (epoch, loss, classification_error))
    else:
        print('Loss of: %s - Classification Error of: %s' % (loss, classification_error))

    return epoch, loss, classification_error


def training(model, train_loader, val_loader, num_epochs, optimizer, loss_function, max_loss_iter, baseline=True,
             exp3=False, initial_model=None):
    print('Training started for %s epochs.' % num_epochs)
    epochs = []
    class_error_list = []
    loss_results = []
    best_loss = 10000  # Picking random high number to assure correct functionality
    loss_iter = 0

    torch.autograd.set_detect_anomaly(True)

    # This rule is made so we can continue learning for experiment 3 - reservoir network population
    # Default is off, which means we start at epoch 0. Else we start at epoch 4
    if not exp3:
        x = range(num_epochs)
    else:
        x = range(P.backprop_epochs-1, num_epochs)
        epochs = initial_model['epoch']
        class_error_list = initial_model['class_error_results']
        loss_results = initial_model['loss_results']

    for epoch in x:

        # Training
        model.train()
        for indx_batch, (batch, targets) in enumerate(train_loader):
            output = model.forward(batch)

            targets = targets.long()

            # Optional print of loss per batch
            # print('Loss in batch %s is: %s' %(indx_batch, loss))

            # Perform back prop after each batch
            loss = loss_function(output, targets)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Perform evaluation after each epoch
        epoch, loss_eval, classification_error = evaluation(val_loader, model, epoch, loss_function)
        epochs.append(epoch)
        class_error_list.append(classification_error)
        loss_results.append(loss_eval)

    dict_results = {
        'model': model,
        'epoch': epochs,
        'loss_results': loss_results,
        'class_error_results': class_error_list,
        'best_loss': best_loss,
        'loss_iter': loss_iter,
    }

    return dict_results
