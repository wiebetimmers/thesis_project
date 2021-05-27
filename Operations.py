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


def get_stats(pop):
    min_list = []
    max_list = []
    mean_list = []
    std_list = []
    for i in range(P.n_epochs):
        epoch_list = []
        for j in range(len(pop)):
            epoch_list.append(pop[j]['loss_results'][i])
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


def combined_plot_exp3(epochs, loss_bl, loss_res, ea_reservoir, border=None, title=''):
    stats_ea = get_stats(ea_reservoir)
    plt.plot(stats_ea['epoch'], stats_ea['mean_list'], 'b-', label='EA Reservoir RNN')
    plt.fill_between(stats_ea['epoch'], stats_ea['min_list'],
                     stats_ea['max_list'], color='b', alpha=0.2)
    plt.plot(epochs, loss_bl, label='Baseline RNN')
    plt.plot(epochs, loss_res, label='Reservoir RNN')

    if border is not None:
        plt.axvline(P.backprop_epochs, label='EA optimizing start', c='r')

    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.legend(loc='upper right')

    if P.mutate_bias:
        bias = 'Bias mutation'
    else:
        bias = 'No bias mutation'

    plt.title(r'$\alpha = %s$, %s' % (P.perturb_rate_decay, bias))
    plt.savefig('plots/exp3/%s.png' % title, bbox_inches='tight')
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
          'mutate_bias: %s\n'
          'sample_dist: %s\n'
          'mu: %s\n'
          'sigma: %s\n'
          'select opt: %s\n'
          'select mech: %s\n'
          'k_best: %s\n'
          'offspring ratio: %s\n'
          'n epochs: %s\n' % (P.population_size, P.generations, P.mutate_opt, P.perturb_rate,
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


def training(model, train_loader, val_loader, num_epochs, optimizer, loss_function, max_loss_iter, baseline=True):
    print('Training started for %s epochs.' % num_epochs)
    epochs = []
    class_error_list = []
    loss_results = []
    best_loss = 10000  # Picking random high number to assure correct functionality
    loss_iter = 0

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):

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
