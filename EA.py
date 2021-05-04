import torch
import numpy as np
import random
import copy
import torch.nn as nn
import Operations as Ops



class EA(object):
    def __init__(self, population_size, val_loader, loss_function, input_size, reservoir_size, n_labels):
        self.population_size = population_size
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = n_labels

    def fitness(self, population, loss_function, parents=None):

        # Copy paste the last results, so we don't have to calculate the loss and accuracy of an unchanged model.
        if parents == True:
            for reservoir in population:
                reservoir['epoch'].append(reservoir['epoch'][-1] + 1)
                reservoir['loss_results'].append(reservoir['loss_results'][-1])
                reservoir['class_error_results'].append(reservoir['class_error_results'][-1])

        else:
            # Evaluate the performance of every (mutated/recombinated) model in the population,
            # add the results to results list.
            for reservoir in population:
                epoch, loss, total_accuracy = Ops.evaluation(self.val_loader,
                                                         reservoir['model'],
                                                         reservoir['epoch'][-1] + 1,
                                                         loss_function)
                reservoir['epoch'].append(epoch)
                reservoir['loss_results'].append(loss)
                reservoir['class_error_results'].append(total_accuracy)

                # If we find a new best model, save it.
                # Still have to fine tune this , make a directory for all the models.
                '''if loss < reservoir['best_loss']:
                    print('* Saving new best model *')
                    torch.save(reservoir['model'], 'trained_reservoir.model')
                    reservoir['best_loss'] = loss
                    reservoir['loss_iter'] = 0
                else:
                    reservoir['loss_iter'] += 1'''

        return population

    def mutation(self, pop, option, offspring_ratio, sample_dist, perturb_rate):
        # Lets pick an offspring ratio of 3 to 1 parent

        if option == 'random_perturbation':
            mut_pop = self.random_perturbation(pop, sample_dist)
            print('Parent / child ratio = 1 : %s' % offspring_ratio)

            if offspring_ratio > 1:
                for i in range(offspring_ratio - 1):
                    mut_pop += self.random_perturbation(pop, sample_dist)

        elif option == 'diff_mutation':
            mut_pop = self.diff_mutation(pop, perturb_rate)

            if offspring_ratio > 1:
                for i in range(offspring_ratio - 1):
                    mut_pop += self.diff_mutation(pop, perturb_rate)

        return mut_pop

    def diff_mutation(self, pop, perturb_rate):
        mut_pop = copy.deepcopy(pop)

        for reservoir in mut_pop:
            # Randomly sample 2 models from the population & split them up
            sample = random.sample(pop, 2)
            sample1 = sample[0]['model']
            sample2 = sample[1]['model']

            # Perturb the weights
            reservoir['model'].layer1.weight += perturb_rate * (sample1.layer1.weight - sample2.layer1.weight)
            reservoir['model'].layer2.weight += perturb_rate * (sample1.layer2.weight - sample2.layer2.weight)
            reservoir['model'].layer3.weight += perturb_rate * (sample1.layer3.weight - sample2.layer3.weight)
            temp_w_out = reservoir['model'].layer4.weight + perturb_rate * (
                        sample1.layer4.weight - sample2.layer4.weight)
            reservoir['model'].layer4.weight = nn.Parameter(temp_w_out, requires_grad=False)

        return mut_pop

    def random_perturbation(self, pop, sample_dist):
        mut_pop = copy.deepcopy(pop)

        for reservoir in mut_pop:
            if sample_dist == 'uniform':
                W_in_sample = torch.empty(self.reservoir_size, self.input_size).uniform_(-0.01, 0.01)
                W_r_sample = torch.empty(self.reservoir_size, self.reservoir_size).uniform_(-0.01, 0.01)
                W_out_sample = torch.empty(self.output_size, self.reservoir_size).uniform_(-0.01, 0.01)
                U_sample = torch.empty(self.reservoir_size, self.input_size).uniform_(-0.01, 0.01)

            elif sample_dist == 'gaussian':
                W_in_sample = torch.empty(self.reservoir_size, self.input_size).normal_(0, 0.05)
                W_r_sample = torch.empty(self.reservoir_size, self.reservoir_size).normal_(0, 0.05)
                W_out_sample = torch.empty(self.output_size, self.reservoir_size).normal_(0, 0.05)
                U_sample = torch.empty(self.reservoir_size, self.input_size).normal_(0, 0.05)

            reservoir['model'].layer1.weight = nn.Parameter(W_in_sample, requires_grad=False)
            reservoir['model'].layer2.weight = nn.Parameter(W_r_sample, requires_grad=False)
            reservoir['model'].layer3.weight = nn.Parameter(U_sample, requires_grad=False)

            # We have to turn off requires grad,
            # because pytorch does not allow inplace mutations on tensors which are used for backprop.
            # See https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/2 .
            reservoir['model'].layer4.weight = nn.Parameter(W_out_sample, requires_grad=False)

        return mut_pop

    def parent_offspring_selection(self, pop, recomb_pop, option):
        # Merge parents and childs
        total_pop = pop + recomb_pop

        # Select the top performing (lowest loss)
        if option == 'loss':
            total_pop = sorted(total_pop, key=lambda k: k['loss_results'][-1])
            new_pop = total_pop[:len(pop)]

        # Select the top performing (lowest classification error)
        elif option == 'classification_error':
            total_pop = sorted(total_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
            new_pop = total_pop[:len(pop)]

        return new_pop

    def keep_best_selection(self, pop, offspring, option, k_best):

        offspring_best = len(pop) - k_best

        if option == 'classification_error':
            # Select the top performing (lowest classification error)
            pop_sorted = sorted(pop, key=lambda k: k['class_error_results'][-1], reverse=False)
            best_pop = pop_sorted[:k_best]

            offspring_sorted = sorted(offspring, key=lambda k: k['class_error_results'][-1], reverse=False)
            best_offspring = offspring_sorted[:offspring_best]

            new_pop = best_pop + best_offspring

        else:
            # Select the top performing (lowest loss)
            pop_sorted = sorted(pop, key=lambda k: k['loss_results'][-1], reverse=False)
            best_pop = pop_sorted[:k_best]

            offspring_sorted = sorted(offspring, key=lambda k: k['loss_results'][-1], reverse=False)
            best_offspring = offspring_sorted[:offspring_best]

            new_pop = best_pop + best_offspring

        return new_pop

    def crossover(self, pop):

        # Using random crossover

        crossed_pop = copy.deepcopy(pop)

        W_in = []
        W_r = []
        U = []
        W_out = []

        # From parent population
        for reservoir in pop:
            W_in.append(reservoir['model'].layer1.weight)
            W_r.append(reservoir['model'].layer2.weight)
            U.append(reservoir['model'].layer3.weight)
            W_out.append(reservoir['model'].layer4.weight)

        # crossover
        for reservoir in crossed_pop:
            reservoir['model'].layer1.weight = random.choice(W_in)
            reservoir['model'].layer3.weight = random.choice(U)
            reservoir['model'].layer2.weight = random.choice(W_r)
            reservoir['model'].layer4.weight = random.choice(W_out)

        return crossed_pop

    def selection(self, pop, offspring, option, select_mech, k_best):

        # Parents + offspring selection
        if select_mech == 'merge_all':
            new_pop = self.parent_offspring_selection(pop, offspring, option)
        elif select_mech == 'keep_k_best':
            new_pop = self.keep_best_selection(pop, offspring, option, k_best)

        return new_pop

    def step(self, pop, mutate_opt, perturb_rate, select_opt, select_mech, offspring_ratio, sample_dist, k_best, loss_function):

        # Apply some mutation and recombination
        mut_pop = self.mutation(pop, mutate_opt, offspring_ratio, sample_dist, perturb_rate)
        crossed_pop = self.crossover(pop)

        # Merge (mutated pop) + ( crossed pop), so we have a larger pool to pick from.
        merged_pop = mut_pop + crossed_pop

        # Get fitness from parents
        pop = self.fitness(pop, loss_function, parents=True)

        # Get fitness from childs
        print('Possible candidates for optimization')
        merged_pop = self.fitness(merged_pop, loss_function, parents=False)

        # Survivor selection
        new_pop = self.selection(pop, merged_pop, select_opt, select_mech, k_best)

        return new_pop