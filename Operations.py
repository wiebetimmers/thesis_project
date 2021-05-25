import matplotlib.pyplot as plt
import torch
import numpy as np
import Datasets as Data
import statistics as st
import Parameters as P

LABELS = Data.get_labels()
plt.rcParams["figure.figsize"] = (10,3)


# Function that transforms the tensor output to a predicted target name.
def categoryFromOutput(output):
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
    print('Best val loss gaussian: %s' %best_gaussian)
    print('Mean Last population gaussian: %s, std: %s' %(stats_gaussian['mean_list'][-1], stats_gaussian['std_list'][-1]))

    plt.plot(stats_uniform['epoch'], stats_uniform['mean_list'], 'r-', label='Uniform')
    plt.fill_between(stats_uniform['epoch'], stats_uniform['min_list'],
                     stats_uniform['max_list'], color='r',
                     alpha=0.2)

    best_uniform = uniform[0]['loss_results'][-1]
    print('Best val loss uniform: %s' % best_uniform)
    print('Mean Last population uniform: %s, std: %s' % (stats_uniform['mean_list'][-1], stats_uniform['std_list'][-1]))

    plt.legend(loc='upper right')
    plt.title(r'$\alpha = %s , \sigma$ = %s' %(P.perturb_rate_decay, P.sigma))
    plt.savefig('plots/exp1/%s.png' %(title), bbox_inches='tight')
    return


def combined_plot_exp3(epochs, loss_bl, loss_res, ea_reservoir, border=None, title=''):
    stats_ea = get_stats(ea_reservoir)
    plt.plot(stats_ea['epoch'], stats_ea['mean_list'], 'b-', label='EA Reservoir RNN')
    plt.fill_between(stats_ea['epoch'], stats_ea['min_list'],
                     stats_ea['max_list'], color='b', alpha=0.2)
    plt.plot(epochs, loss_bl, label='Baseline RNN')
    plt.plot(epochs, loss_res, label='Reservoir RNN')

    if border != None:
        plt.axvline(P.backprop_epochs, label='EA optimizing start', c='r')

    plt.legend(loc='upper right')
    plt.title('')
    plt.savefig('plots/exp3/%s.png' % (title), bbox_inches='tight')
    return


# Concatenating the results of all batches in the lists, calculating the total accuracy.
def accuracy(pred_targets_list, gold_targets_list):
    total_correct = 0
    total_amount = 0

    zip_list = zip(pred_targets_list, gold_targets_list)
    for pred_targets, gold_targets in zip_list:
        total_correct += (pred_targets == gold_targets).float().sum()
        total_amount += len(pred_targets)

    accuracy = 100 * total_correct / total_amount

    return accuracy.item()


# Concatenating the results of all batches in the lists, calculating the classification error.
def class_error(pred_targets_list, gold_targets_list):
    total_error = 0
    total_amount = 0

    zip_list = zip(pred_targets_list, gold_targets_list)
    for pred_targets, gold_targets in zip_list:
        total_error += (pred_targets != gold_targets).float().sum()
        total_amount += len(pred_targets)

    class_error = (total_error / total_amount) * 100

    return class_error.item()


# Evaluation -> used for validation and test set.
def evaluation(val_loader, model, epoch, loss_function):
    # Evaluating our performance so far
    model.eval()

    # Store all results in a list to calculate the accuracy.
    pred_target_total_acc = []
    target_total_acc = []

    # Initialize counters / c
    loss = 0.
    N = 0.

    # Iterating over the validation set batches, acquiring tensor formatted results.
    for indx_batch, (batch, targets) in enumerate(val_loader):
        output = model.forward(batch)
        pred_targets = np.array([])
        for item in output:
            pred_targets = np.append(pred_targets, categoryFromOutput(item))
        pred_targets = torch.from_numpy(pred_targets).int()

        # Calculating loss
        loss_t = loss_function(output, targets.long())
        loss = loss + loss_t.item()
        N = N + batch.shape[0]

        # Append the batch result to a list of all results
        pred_target_total_acc.append(pred_targets)
        target_total_acc.append(targets)

    # Store the loss corrected by its size
    loss = loss / N

    classification_error = class_error(pred_target_total_acc, target_total_acc)
    print('Epoch: %s - Loss of: %s - Classification Error of: %s' % (epoch, loss, classification_error))

    return epoch, loss, classification_error


def baseline_control(epoch, model, best_loss, loss_eval, loss_iter, max_loss_iter):
    if epoch == 0:
        # print('* Saving 1st epoch model *')
        # torch.save(model, 'trained_baseline.model')
        best_loss = loss_eval
    else:
        if loss_eval < best_loss:
            # print('* Saving new best model *')
            # torch.save(model, 'trained_baseline.model')
            best_loss = loss_eval
            loss_iter = 0
        else:
            loss_iter += 1

    # If loss has not improved for an arbitrary amount of epochs:
    # if loss_iter > max_loss_iter:

    return best_loss, loss_iter


def training(model, train_loader, val_loader, num_epochs, optimizer, loss_function, max_loss_iter, baseline=True):
    print('Training started for %s epochs.' % (num_epochs))
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

            loss = loss_function(output, targets)

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

        if baseline == True:
            best_loss, loss_iter = baseline_control(epoch, model, best_loss, loss_eval, loss_iter, max_loss_iter)

    dict_results = {
        'model': model,
        'epoch': epochs,
        'loss_results': loss_results,
        'class_error_results': class_error_list,
        'best_loss': best_loss,
        'loss_iter': loss_iter,
    }

    return dict_results
