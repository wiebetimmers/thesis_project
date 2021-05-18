import matplotlib.pyplot as plt
import torch
import numpy as np
import Datasets as Data

LABELS = Data.get_labels()
plt.rcParams["figure.figsize"] = (20,3)

# Function that transforms the tensor output to a predicted target name.
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return LABELS[category_i]


# Plot both accuracy as log loss.
def plot_results(epochs, loss, class_error, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%s' % title)
    ax1.set(ylabel='Loss')
    ax2.set(ylabel='Classification Error', xlabel='Epochs')

    ax1.plot(epochs, loss)
    ax2.plot(epochs, class_error)
    plt.savefig('%s plot.png' % (title), bbox_inches='tight')

    return


def combined_plot_result(epochs, loss_bl, class_error_bl,
                         loss_res, class_error_res,
                         loss_evo, class_error_evo,
                         border=None,
                         label_bl='', label_res='', label_evo='', title=''):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%s' % title)
    ax1.set(ylabel='Loss', xlabel='Epochs')
    ax2.set(ylabel='Classification Error', xlabel='Epochs')

    ax1.plot(epochs, loss_bl, label=label_bl)
    ax1.plot(epochs, loss_res, label=label_res)
    ax1.plot(epochs, loss_evo, label=label_evo)
    ax2.plot(epochs, class_error_bl, label=label_bl)
    ax2.plot(epochs, class_error_res, label=label_res)
    ax2.plot(epochs, class_error_evo, label=label_evo)

    if border != None:
        ax1.axvline(border, label='EA optimizing start', c='r')
        ax2.axvline(border, label='EA optimizing start', c='r')

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    plt.savefig('plots/%s.png' % (title), bbox_inches='tight')
    return

def best_pop_plot(best_pop, best_individual, title):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%s' % title)
    ax1.set(ylabel='Loss')
    ax2.set(ylabel='Classification Error', xlabel='Epochs')

    for model in best_pop:
        ax1.plot(model['epoch'], model['loss_results'], c='b')
        ax2.plot(model['epoch'], model['class_error_results'], c='b')

    ax1.plot(best_individual['epoch'], best_individual['loss_results'], c='r', label='Best individual')
    ax2.plot(best_individual['epoch'], best_individual['class_error_results'], c='r', label='Best individual')

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.savefig('plots/%s.png' % (title), bbox_inches='tight')

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


def combined_plot_result_distributions(epochs, loss_gaussian, class_error_gaussian,
                                       loss_uniform, class_error_uniform,
                                       loss_cauchy, class_error_cauchy,
                                       loss_lognormal, class_error_lognormal,
                                       border=None,
                                       label_gaussian='', label_uniform='', label_cauchy='', label_lognormal='',
                                       title=''):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('%s' % title)
    ax1.set(ylabel='Loss', xlabel='Epochs')
    ax2.set(ylabel='Classification Error', xlabel='Epochs')

    ax1.plot(epochs, loss_gaussian, label=label_gaussian)
    ax1.plot(epochs, loss_uniform, label=label_uniform)
    ax1.plot(epochs, loss_cauchy, label=label_cauchy)
    ax1.plot(epochs, loss_lognormal, label=label_lognormal)
    ax2.plot(epochs, class_error_gaussian, label=label_gaussian)
    ax2.plot(epochs, class_error_uniform, label=label_uniform)
    ax2.plot(epochs, class_error_cauchy, label=label_cauchy)
    ax2.plot(epochs, class_error_lognormal, label=label_lognormal)

    if border != None:
        ax1.axvline(border, label='EA optimizing start', c='r')
        ax2.axvline(border, label='EA optimizing start', c='r')

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    plt.savefig('plots/%s.png' % (title), bbox_inches='tight')
    return