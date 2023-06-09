import gc
import getopt
import math
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

import utilities
import GreedyAlgorithm
import AntColony
import branchAndBound as bnb
from data_generator import generate_compatibility_matrix


def average(
        array: list):
    return sum(array) / len(array)


def ant_colony_testing(
    iterations: int,
    task_size: int,
    alfa_pheromone_influence=-1,
    beta_data_influence=-1,
    p_evaporation_coeficient=-1,
    gamma_number_of_iterations=-1,
    theta_number_of_ants_per_vertice=-1,

):
    size = task_size
    variants = iterations

    concat_params = [
        alfa_pheromone_influence,
        beta_data_influence,
        p_evaporation_coeficient,
        gamma_number_of_iterations,
        theta_number_of_ants_per_vertice
    ]

    alfa_testing_data = []
    beta_testing_data = []
    p_testing_data = []
    gamma_testing_data = []
    theta_testing_data = []

    is_given_by_user_flags = {
        "alfa": False if alfa_pheromone_influence == -1 else True,
        "beta": False if beta_data_influence == -1 else True,
        "p": False if p_evaporation_coeficient == -1 else True,
        "gamma": False if gamma_number_of_iterations == -1 else True,
        "theta": False if theta_number_of_ants_per_vertice == -1 else True,
    }

    how_much_to_choose = {
        'a': 1, 'b': 2, 'c': 3
    }
    # comment out permutations to speed up iterations
    # all_possible_choosings = [i for i in permutations([1, 2, 3], 3)]
    all_possible_choosings = [
        (how_much_to_choose["a"], how_much_to_choose["b"], how_much_to_choose["c"])]

    # place defaults if no param was given
    if alfa_pheromone_influence == -1:
        alfa_pheromone_influence = 0.8
        alfa_testing_data = [x for x in np.arange(0.1, 2, 0.1)]

    if beta_data_influence == -1:
        beta_data_influence = 1.5
        beta_testing_data = [x for x in np.arange(0.1, 2, 0.1)]

    if p_evaporation_coeficient == -1:
        p_evaporation_coeficient = 0.3
        p_testing_data = [x for x in np.arange(0.1, 1, 0.1)]

    if gamma_number_of_iterations == -1:
        gamma_number_of_iterations = 20
        gamma_testing_data = [x for x in np.arange(1, 20, 1)]

    if theta_number_of_ants_per_vertice == -1:
        theta_number_of_ants_per_vertice = 1
        theta_testing_data = [1, 2, 3, 4]

    x_axis = [[], []]
    y_axis = []

    def PlotData():
        fig1, ax1 = plt.subplots()
        ax1.plot(x_axis[0], y_axis, "-bh", label="ant colony")
        plt.legend(loc="upper left")
        plt.title(
            "Залежність точності від параметру мурашиного алгоритму")
        plt.xlabel(x_axis[1])
        plt.ylabel("загальна взаємопридатність, од")
        # plt.ylim(10, 25)

    for param, flag in is_given_by_user_flags.items():
        if param == "alfa" and not flag:
            x_axis = [[], []]
            y_axis = []

            x_axis[0] = alfa_testing_data
            x_axis[1] = "параметр контролю впливу феромону на рішення мурахи"

            for jjjj in alfa_testing_data:
                collected_corr_values = []

                for k in range(variants):
                    data = generate_compatibility_matrix(size, size, size, '')
                    total_corr_value = 0
                    for i in range(gamma_number_of_iterations):
                        print("alfa$ x value = ", jjjj,
                              ", iteration = ", i,  ", variant=", k)
                        for choo in all_possible_choosings:
                            group_indices = utilities.get_group_indices(data)
                            pheromone = np.random.rand(
                                data.shape[0], data.shape[1])
                            how_much_to_choose = {
                                'a': choo[0], 'b': choo[1], 'c': choo[2]
                            }

                            ant_paths = AntColony.ant_iteration(
                                data, group_indices, how_much_to_choose, pheromone,
                                theta_number_of_ants_per_vertice,
                                beta_data_influence, jjjj
                            )

                            AntColony.update_pheromone_array(
                                data, group_indices, how_much_to_choose, pheromone,
                                ant_paths, p_evaporation_coeficient
                            )

                            for ant, values in ant_paths[-1].items():
                                values['value'] = bnb.bound_from_string(
                                    data, group_indices, how_much_to_choose, ant)

                            _, iter_max_ant = bnb.data_dict_to_treedict_converter(
                                [ant_paths[-1]])

                            for key, values in iter_max_ant.items():
                                if values['value'] > total_corr_value:
                                    total_corr_value = values['value']
                    collected_corr_values.append(total_corr_value)
                y_axis.append(average(collected_corr_values))
            PlotData()

        if param == "beta" and not flag:
            x_axis = [[], []]
            y_axis = []

            x_axis[0] = beta_testing_data
            x_axis[1] = "параметр контролю впливу взаємопридатності на рішення мурахи"

            for jjjj in beta_testing_data:
                collected_corr_values = []

                for k in range(variants):
                    data = generate_compatibility_matrix(size, size, size, '')
                    total_corr_value = 0

                    for i in range(gamma_number_of_iterations):
                        print("beta$ x value = ", jjjj,
                              ", iteration = ", i,  ", variant=", k)
                        for choo in all_possible_choosings:
                            group_indices = utilities.get_group_indices(data)
                            pheromone = np.random.rand(
                                data.shape[0], data.shape[1])
                            how_much_to_choose = {
                                'a': choo[0], 'b': choo[1], 'c': choo[2]
                            }

                            ant_paths = AntColony.ant_iteration(
                                data, group_indices, how_much_to_choose, pheromone,
                                theta_number_of_ants_per_vertice,
                                jjjj, alfa_pheromone_influence
                            )

                            AntColony.update_pheromone_array(
                                data, group_indices, how_much_to_choose, pheromone,
                                ant_paths, p_evaporation_coeficient
                            )

                            for ant, values in ant_paths[-1].items():
                                values['value'] = bnb.bound_from_string(
                                    data, group_indices, how_much_to_choose, ant)

                            _, iter_max_ant = bnb.data_dict_to_treedict_converter(
                                [ant_paths[-1]])

                            for key, values in iter_max_ant.items():
                                if values['value'] > total_corr_value:
                                    total_corr_value = values['value']
                    collected_corr_values.append(total_corr_value)
                y_axis.append(average(collected_corr_values))
            PlotData()

        if param == "p" and not flag:
            x_axis = [[], []]
            y_axis = []

            x_axis[0] = p_testing_data
            x_axis[1] = "стале випаровування феромону"

            for jjjj in p_testing_data:
                collected_corr_values = []

                for k in range(variants):
                    data = generate_compatibility_matrix(size, size, size, '')
                    total_corr_value = 0
                    for i in range(gamma_number_of_iterations):
                        print("p$ x value = ", jjjj,
                              ", iteration = ", i,  ", variant=", k)
                        for choo in all_possible_choosings:
                            group_indices = utilities.get_group_indices(data)
                            pheromone = np.random.rand(
                                data.shape[0], data.shape[1])
                            how_much_to_choose = {
                                'a': choo[0], 'b': choo[1], 'c': choo[2]
                            }

                            ant_paths = AntColony.ant_iteration(
                                data, group_indices, how_much_to_choose, pheromone,
                                theta_number_of_ants_per_vertice,
                                beta_data_influence, alfa_pheromone_influence
                            )

                            AntColony.update_pheromone_array(
                                data, group_indices, how_much_to_choose, pheromone,
                                ant_paths, jjjj
                            )

                            for ant, values in ant_paths[-1].items():
                                values['value'] = bnb.bound_from_string(
                                    data, group_indices, how_much_to_choose, ant)

                            _, iter_max_ant = bnb.data_dict_to_treedict_converter(
                                [ant_paths[-1]])

                            for key, values in iter_max_ant.items():
                                if values['value'] > total_corr_value:
                                    total_corr_value = values['value']
                    collected_corr_values.append(total_corr_value)
                y_axis.append(average(collected_corr_values))
            PlotData()

        if param == "gamma" and not flag:
            x_axis = [[], []]
            y_axis = []

            x_axis[0] = gamma_testing_data
            x_axis[1] = "кількість ітерацій алгоритму"

            for jjjj in gamma_testing_data:
                collected_corr_values = []

                for k in range(variants):
                    data = generate_compatibility_matrix(size, size, size, '')
                    total_corr_value = 0

                    for i in range(jjjj):
                        print("gamma$ x value=", jjjj, ", iteration=",
                              jjjj, ", variant=", k)
                        for choo in all_possible_choosings:
                            group_indices = utilities.get_group_indices(data)
                            pheromone = np.random.rand(
                                data.shape[0], data.shape[1])
                            how_much_to_choose = {
                                'a': choo[0], 'b': choo[1], 'c': choo[2]
                            }

                            ant_paths = AntColony.ant_iteration(
                                data, group_indices, how_much_to_choose, pheromone,
                                theta_number_of_ants_per_vertice,
                                beta_data_influence, alfa_pheromone_influence
                            )

                            AntColony.update_pheromone_array(
                                data, group_indices, how_much_to_choose, pheromone,
                                ant_paths, p_evaporation_coeficient
                            )

                            for ant, values in ant_paths[-1].items():
                                values['value'] = bnb.bound_from_string(
                                    data, group_indices, how_much_to_choose, ant)

                            _, iter_max_ant = bnb.data_dict_to_treedict_converter(
                                [ant_paths[-1]])

                            for key, values in iter_max_ant.items():
                                if values['value'] > total_corr_value:
                                    total_corr_value = values['value']
                    collected_corr_values.append(total_corr_value)
                y_axis.append(average(collected_corr_values))
            PlotData()

        if param == "theta" and not flag:
            x_axis = [[], []]
            y_axis = []

            x_axis[0] = theta_testing_data
            x_axis[1] = "кількість мурах на розставлених на вершинах в першій ітерації"

            for jjjj in theta_testing_data:
                collected_corr_values = []
                for k in range(variants):
                    data = generate_compatibility_matrix(size, size, size, '')
                    total_corr_value = 0

                    for i in range(gamma_number_of_iterations):
                        print("theta$ x value = ", jjjj, ", iteration = ", i)
                        for choo in all_possible_choosings:
                            group_indices = utilities.get_group_indices(data)
                            pheromone = np.random.rand(
                                data.shape[0], data.shape[1])
                            how_much_to_choose = {
                                'a': choo[0], 'b': choo[1], 'c': choo[2]
                            }

                            ant_paths = AntColony.ant_iteration(
                                data, group_indices, how_much_to_choose, pheromone,
                                jjjj,
                                beta_data_influence, alfa_pheromone_influence
                            )

                            AntColony.update_pheromone_array(
                                data, group_indices, how_much_to_choose, pheromone,
                                ant_paths, p_evaporation_coeficient
                            )

                            for ant, values in ant_paths[-1].items():
                                values['value'] = bnb.bound_from_string(
                                    data, group_indices, how_much_to_choose, ant)

                            _, iter_max_ant = bnb.data_dict_to_treedict_converter(
                                [ant_paths[-1]])

                            for key, values in iter_max_ant.items():
                                if values['value'] > total_corr_value:
                                    total_corr_value = values['value']
                    collected_corr_values.append(total_corr_value)
                y_axis.append(average(collected_corr_values))
            PlotData()
    plt.show()


def main():
    whole_iterations = 1
    task_size = 5
    alfa_pheromone_influence = -1
    beta_data_influence = -1
    p_evaporation_coeficient = -1
    gamma_number_of_iterations = -1
    theta_number_of_ants_per_vertice = -1
    output_path = ''

    opts, args = getopt.getopt(sys.argv[1:], "h",
                               ["iter=",
                                "ts=",
                                "api=",
                                "bdi=",
                                "pec=",
                                "gni=",
                                "tnav=",
                                "output_path="])

    for opt, arg in opts:
        if opt == '-h':
            print(
                'python overall_time_testing.py [--api=<_>]'
                ' [--bdi=<_>] [--pec=<_>]'
                ' [--gni=<_] [--tnav=]')
            sys.exit()
        elif opt in "--iter=":
            whole_iterations = int(arg)
        elif opt in "--ts=":
            task_size = int(arg)
        elif opt in "--api=":
            alfa_pheromone_influence = float(arg)
        elif opt in "--bdi=":
            beta_data_influence = float(arg)
        elif opt in "--pec=":
            p_evaporation_coeficient = float(arg)
        elif opt in "--gni=":
            gamma_number_of_iterations = int(arg)
        elif opt in "--tnav=":
            theta_number_of_ants_per_vertice = int(arg)
        elif opt in "--output_path=":
            output_path = arg

    # ant_colony_testing(-1, 0.1, 0.3, 10, 1)
    # ant_colony_testing(0.8, -1, 0.3, 10, 1)
    # ant_colony_testing(0.8, 0.1, -1, 10, 1)
    # ant_colony_testing(0.8, 0.1, 0.3, -1, 1)
    # ant_colony_testing(0.8, 0.1, 0.3, 10, -1)
    ant_colony_testing(whole_iterations,
                       task_size,
                       alfa_pheromone_influence,
                       beta_data_influence,
                       p_evaporation_coeficient,
                       gamma_number_of_iterations,
                       theta_number_of_ants_per_vertice)


if __name__ == "__main__":
    main()
