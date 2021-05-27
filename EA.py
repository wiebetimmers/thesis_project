import torch
import numpy as np
import random
import copy
import torch.nn as nn
import Operations as Ops
import Parameters as P


class EA(object):
    def __init__(self, population_size, val_loader, loss_function, input_size, reservoir_size, n_labels):
        self.population_size = population_size
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = n_labels

    def fitness(self, population, parents=None):

        # Copy paste the last results, so we don't have to calculate the loss and accuracy of an unchanged model.
        if parents:
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
                                                         P.loss_function)
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

    def mutation(self, pop, perturb_rate):

        if P.mutate_opt == 'random_perturbation':
            mut_pop = self.random_perturbation(pop, P.sample_dist, P.mutate_bias, P.mu, P.sigma, perturb_rate)
        else:  # 'diff_mutation'
            mut_pop = self.diff_mutation(pop, perturb_rate)

        return mut_pop

    def sample_two(self, pop):
        sample = random.sample(pop, 2)
        sample1 = sample[0]['model']
        sample2 = sample[1]['model']
        return sample1, sample2

    def diff_mutation(self, pop, perturb_rate):
        mut_pop = copy.deepcopy(pop)

        for reservoir in mut_pop:
            # Randomly sample 2 models from the population & split them up
            sample1, sample2 = self.sample_two(pop)

            # Perturb the weights
            reservoir['model'].layer1.weight += perturb_rate * (sample1.layer1.weight - sample2.layer1.weight)
            reservoir['model'].layer2.weight += perturb_rate * (sample1.layer2.weight - sample2.layer2.weight)
            reservoir['model'].layer3.weight += perturb_rate * (sample1.layer3.weight - sample2.layer3.weight)
            temp_w_out = reservoir['model'].layer4.weight + perturb_rate * (
                        sample1.layer4.weight - sample2.layer4.weight)
            reservoir['model'].layer4.weight = nn.Parameter(temp_w_out, requires_grad=False)

            if P.mutate_bias:
                # Perturb the bias
                reservoir['model'].layer1.bias += perturb_rate * (sample1.layer1.bias - sample2.layer1.bias)
                reservoir['model'].layer2.bias += perturb_rate * (sample1.layer2.bias - sample2.layer2.bias)
                reservoir['model'].layer3.bias += perturb_rate * (sample1.layer3.bias - sample2.layer3.bias)
                temp_w_out = reservoir['model'].layer4.bias + perturb_rate * (
                        sample1.layer4.bias - sample2.layer4.bias)
                reservoir['model'].layer4.bias = nn.Parameter(temp_w_out, requires_grad=False)

        return mut_pop

    def random_perturbation(self, pop, sample_dist, mutate_bias, mu, sigma, perturb_rate):
        mut_pop = copy.deepcopy(pop)

        for reservoir in mut_pop:

            # sample values
            if sample_dist == 'uniform':
                W_in_sample = torch.empty(self.reservoir_size, self.input_size).uniform_(-3*P.sigma, 3*P.sigma)
                W_r_sample = torch.empty(self.reservoir_size, self.reservoir_size).uniform_(-3*P.sigma, 3*P.sigma)
                W_out_sample = torch.empty(self.output_size, self.reservoir_size).uniform_(-3*P.sigma, 3*P.sigma)
                U_sample = torch.empty(self.reservoir_size, self.input_size).uniform_(-3*P.sigma, 3*P.sigma)

                if mutate_bias:
                    W_in_bias = torch.empty(1, self.reservoir_size).uniform_(-3*P.sigma, 3*P.sigma)
                    W_r_bias = torch.empty(1, self.reservoir_size).uniform_(-3*P.sigma, 3*P.sigma)
                    W_out_bias = torch.empty(1, self.output_size).uniform_(-3*P.sigma, 3*P.sigma)
                    U_bias = torch.empty(1, self.reservoir_size).uniform_(-3*P.sigma, 3*P.sigma)

            elif sample_dist == 'gaussian':
                W_in_sample = torch.empty(self.reservoir_size, self.input_size).normal_(mu, sigma)
                W_r_sample = torch.empty(self.reservoir_size, self.reservoir_size).normal_(mu, sigma)
                W_out_sample = torch.empty(self.output_size, self.reservoir_size).normal_(mu, sigma)
                U_sample = torch.empty(self.reservoir_size, self.input_size).normal_(mu, sigma)

                if mutate_bias:
                    W_in_bias = torch.empty(1, self.reservoir_size).normal_(mu, sigma)
                    W_r_bias = torch.empty(1, self.reservoir_size).normal_(mu, sigma)
                    W_out_bias = torch.empty(1, self.output_size).normal_(mu, sigma)
                    U_bias = torch.empty(1, self.reservoir_size).normal_(mu, sigma)

            elif sample_dist == 'cauchy':
                W_in_sample = torch.empty(self.reservoir_size, self.input_size).cauchy_(mu, sigma)
                W_r_sample = torch.empty(self.reservoir_size, self.reservoir_size).cauchy_(mu, sigma)
                W_out_sample = torch.empty(self.output_size, self.reservoir_size).cauchy_(mu, sigma)
                U_sample = torch.empty(self.reservoir_size, self.input_size).cauchy_(mu, sigma)

                if mutate_bias:
                    W_in_bias = torch.empty(1, self.reservoir_size).cauchy_(mu, sigma)
                    W_r_bias = torch.empty(1, self.reservoir_size).cauchy_(mu, sigma)
                    W_out_bias = torch.empty(1, self.output_size).cauchy_(mu, sigma)
                    U_bias = torch.empty(1, self.reservoir_size).cauchy_(mu, sigma)

            elif sample_dist == 'lognormal':
                W_in_sample = torch.empty(self.reservoir_size, self.input_size).log_normal_(mu, sigma)
                W_r_sample = torch.empty(self.reservoir_size, self.reservoir_size).log_normal_(mu, sigma)
                W_out_sample = torch.empty(self.output_size, self.reservoir_size).log_normal_(mu, sigma)
                U_sample = torch.empty(self.reservoir_size, self.input_size).log_normal_(mu, sigma)

                if mutate_bias:
                    W_in_bias = torch.empty(1, self.reservoir_size).log_normal_(mu, sigma)
                    W_r_bias = torch.empty(1, self.reservoir_size).log_normal_(mu, sigma)
                    W_out_bias = torch.empty(1, self.output_size).log_normal_(mu, sigma)
                    U_bias = torch.empty(1, self.reservoir_size).log_normal_(mu, sigma)

            # Add sampled value to the current weights, adjusted by the perturb rate.
            reservoir['model'].layer1.weight = nn.Parameter((perturb_rate*W_in_sample)+reservoir['model'].layer1.weight,
                                                            requires_grad=False)
            reservoir['model'].layer2.weight = nn.Parameter((perturb_rate*W_r_sample)+reservoir['model'].layer2.weight,
                                                            requires_grad=False)
            reservoir['model'].layer3.weight = nn.Parameter((perturb_rate*U_sample)+reservoir['model'].layer3.weight,
                                                            requires_grad=False)
            reservoir['model'].layer4.weight = nn.Parameter((perturb_rate*W_out_sample)+reservoir['model'].layer4.weight,
                                                            requires_grad=False)

            # Add sampled value to the current bias weights, adjusted by the perturb rate.
            if mutate_bias:
                reservoir['model'].layer1.bias = nn.Parameter((perturb_rate*W_in_bias)+reservoir['model'].layer1.bias,
                                                              requires_grad=False)
                reservoir['model'].layer2.bias = nn.Parameter((perturb_rate*W_r_bias)+reservoir['model'].layer2.bias,
                                                              requires_grad=False)
                reservoir['model'].layer3.bias = nn.Parameter((perturb_rate*U_bias)+reservoir['model'].layer3.bias,
                                                              requires_grad=False)
                reservoir['model'].layer4.bias = nn.Parameter((perturb_rate*W_out_bias)+reservoir['model'].layer4.bias,
                                                              requires_grad=False)

        return mut_pop

    def merge_all_selection(self, pop, recomb_pop):
        # Merge parents and childs
        total_pop = pop + recomb_pop

        # Select the top performing (lowest loss)
        if P.select_opt == 'loss':
            total_pop = sorted(total_pop, key=lambda k: k['loss_results'][-1])
            new_pop = total_pop[:len(pop)]

        # Select the top performing (lowest classification error)
        else:  # 'classification_error'
            total_pop = sorted(total_pop, key=lambda k: k['class_error_results'][-1], reverse=False)
            new_pop = total_pop[:len(pop)]

        return new_pop

    def keep_best_selection(self, pop, offspring):

        offspring_best = len(pop) - P.k_best

        if P.select_opt == 'classification_error':
            value = 'class_error_results'
        else:
            value = 'loss_results'

        # Select the top performing (lowest classification error or lowest loss)
        pop_sorted = sorted(pop, key=lambda k: k[value][-1], reverse=False)
        best_pop = pop_sorted[:P.k_best]

        offspring_sorted = sorted(offspring, key=lambda k: k[value][-1], reverse=False)
        best_offspring = offspring_sorted[:offspring_best]

        new_pop = best_pop + best_offspring

        return new_pop

    def keep_best_offspring(self, pop, offspring):

        if P.select_opt == 'classification_error':
            value = 'class_error_results'
        else:
            value = 'loss_results'

        # Select the best performing offspring
        offspring_sorted = sorted(offspring, key=lambda k: k[value][-1], reverse=False)
        best_offspring = offspring_sorted[:len(pop)]

        return best_offspring

    def crossover(self, pop):

        # Using random crossover

        crossed_pop = copy.deepcopy(pop)

        W_in = []
        W_r = []
        U = []
        W_out = []

        if P.mutate_bias:
            W_in_bias = []
            W_r_bias = []
            U_bias = []
            W_out_bias = []

        # From parent population
        for reservoir in pop:
            W_in.append(reservoir['model'].layer1.weight)
            W_r.append(reservoir['model'].layer2.weight)
            U.append(reservoir['model'].layer3.weight)
            W_out.append(reservoir['model'].layer4.weight)

            if P.mutate_bias:
                W_in_bias.append(reservoir['model'].layer1.bias)
                W_r_bias.append(reservoir['model'].layer2.bias)
                U_bias.append(reservoir['model'].layer3.bias)
                W_out_bias.append(reservoir['model'].layer4.bias)

        # crossover
        for reservoir in crossed_pop:
            reservoir['model'].layer1.weight = random.choice(W_in)
            reservoir['model'].layer3.weight = random.choice(U)
            reservoir['model'].layer2.weight = random.choice(W_r)
            reservoir['model'].layer4.weight = random.choice(W_out)

            if P.mutate_bias:
                reservoir['model'].layer1.bias = random.choice(W_in_bias)
                reservoir['model'].layer3.bias = random.choice(U_bias)
                reservoir['model'].layer2.bias = random.choice(W_r_bias)
                reservoir['model'].layer4.bias = random.choice(W_out_bias)

        return crossed_pop

    def selection(self, pop, offspring):

        # Parents + offspring selection
        if P.select_mech == 'merge_all':
            new_pop = self.merge_all_selection(pop, offspring)
        elif P.select_mech == 'keep_k_best_parents':
            new_pop = self.keep_best_selection(pop, offspring)
        else:  # 'keep_best_offspring'
            new_pop = self.keep_best_offspring(pop, offspring)

        return new_pop

    def dynamic_perturb_rate(self, epoch):  # Perturb rate decay
        new_rate = P.perturb_rate * (1.0 / (1.0 + P.perturb_rate_decay * epoch))
        return new_rate

    def step(self, pop, epoch):

        perturb_rate = self.dynamic_perturb_rate(epoch)
        print('Perturb rate: %s, see below for mutated and crossed over population' % perturb_rate)

        # Apply some mutation and recombination
        mut_pop = self.mutation(pop, perturb_rate)
        crossed_pop = self.crossover(pop)
        mut_crossed_pop = self.mutation(crossed_pop, perturb_rate)

        # Optional: offspring ratio to increase offspring size.
        if P.offspring_ratio > 1:
            for i in range(P.offspring_ratio - 1):
                mut_pop += self.mutation(pop, perturb_rate)
                crossed_pop += self.crossover(pop)
                mut_crossed_pop += self.mutation(crossed_pop, perturb_rate)

        # Merge (mutated pop) + ( crossed pop) + (mutated & crossed), so we have a large offspring pool to pick from.
        merged_pop = mut_pop + mut_crossed_pop + crossed_pop  # + crossed_pop

        # Get fitness from parents
        pop = self.fitness(pop, parents=True)

        # Get fitness from childs
        merged_pop = self.fitness(merged_pop, parents=False)

        # Survivor selection
        new_pop = self.selection(pop, merged_pop)

        return new_pop
