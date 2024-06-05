import pandas as pd
from causalnex.structure import dynotears as dnt
import time
import numpy as np
from utils import *
import random


def train(files, graph_file, results_file, lambda_a=None, n_trials=10, measurements=None):
    data = [pd.read_table(path + os.sep + file, header=None, index_col=0).transpose() for file in files]
    variables = data[0].columns
    # for i in range(100) : print(files[i] + " " + str(len((data[i].columns))))

    gts = read_graph(path + os.sep + "groundtruth.txt")

    # forbid edges that are within the same time slice
    tabu_edges = [(0, u, v) for u in variables for v in variables]

    if lambda_a is None:
        lambda_a = gsearch_lambda(data, tabu_edges,
                                  [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000,
                                   1000000], gts)
    print("Using lambda_a: " + str(lambda_a))

    #TODO refactor pandas DF
    elapsed_times = []
    elapsed_times_std = []
    edit_distances = []
    edit_distances_std = []
    f1_scores = []
    f1_scores_std = []
    f1_leave_out = []
    f1_leave_out_std = []
    if measurements is None:
        measurements = range(1, len(data) + 1)
    for n in measurements:
        # create DBN using dynotears algorithm from the causalnex package
        # start_time = time.time()
        # dbn = dnt.from_pandas_dynamic(time_series=data[:n], p=1, tabu_edges=tabu_edges, lambda_a=lambda_a)
        # elapsed_times.append(time.time() - start_time)
        #
        # # now compare with the ground truth graph
        # dbns = set(dbn.edges)
        # edit_distances.append(len(gts.union(dbns)) - len(gts.intersection(dbns)))
        # f1_scores.append(f1score(dbns, gts))

        # TODO refactor w data frame
        time, time_std, ed, ed_std, f1, f1_std, f1_lo, f1_lo_std, dbns = train_single(data[:n], tabu_edges, lambda_a, gts, n_trials)
        elapsed_times.append(time)
        elapsed_times_std.append(time_std)
        edit_distances.append(ed)
        edit_distances_std.append(ed_std)
        f1_scores.append(f1)
        f1_scores_std.append(f1_std)
        f1_leave_out.append(f1_lo)
        f1_leave_out_std.append(f1_lo_std)
        print("Training  n: " + str(n))

        if n is len(data):
            write_graph(dbns, graph_file)
    # visualization https://github.com/mckinsey/causalnex/issues/74

    with open(results_file, "w") as file:
        file.write("Nseries\tTime\tTimestd\tAcc\tAccstd\tF1\tF1std\tF1LO\tF1LOstd\n")
        for i in range(len(elapsed_times)):
            file.write(str(i + 1) + "\t" + str(elapsed_times[i]) + "\t" + str(elapsed_times_std[i]) + "\t" +
                       str(edit_distances[i]) + "\t" + str(edit_distances_std[i]) + "\t" + str(f1_scores[i]) +
                       "\t" + str(f1_scores_std[i]) + "\t" + str(f1_leave_out[i]) + "\t" + str(f1_leave_out_std[i]) + "\n")


def train_single(data, tabu_edges, lambda_a, gts, n_trials):
    times = np.zeros(n_trials)
    edit_distances = np.zeros(n_trials)
    f1_scores = np.zeros(n_trials)
    f1lo = np.zeros(n_trials)
    for i in range(n_trials):
        start_time = time.time()
        dbn = dnt.from_pandas_dynamic(time_series=data, p=1, tabu_edges=tabu_edges, lambda_a=lambda_a)
        times[i] = time.time() - start_time

        # now compare with the ground truth graph
        dbns = set(dbn.edges)
        edit_distances[i] = len(gts.union(dbns)) - len(gts.intersection(dbns))
        f1_scores[i] = f1score(dbns, gts)

    for i in range(n_trials):
        sample_size = max(min(int(0.9 * len(data)), len(data) - 1), 1)
        sample = random.sample(data, sample_size)
        dbn = dnt.from_pandas_dynamic(time_series=sample, p=1, tabu_edges=tabu_edges, lambda_a=lambda_a)
        dbnslo = set(dbn.edges)
        f1lo[i] = f1score(dbnslo, gts)

    return np.mean(times), np.std(times), np.mean(edit_distances), np.std(edit_distances), np.mean(f1_scores), np.std(
        f1_scores), np.mean(f1lo), np.std(f1lo), dbns


def gsearch_lambda(data, tabu_edges, lambda_as, gts):
    maximum = 0.0
    lambda_a_max = 0
    for lambda_a in lambda_as:
        dbn = dnt.from_pandas_dynamic(time_series=data, p=1, tabu_edges=tabu_edges, lambda_a=lambda_a)
        dbns = set(dbn.edges)
        f1_score = f1score(dbns, gts)
        print("lambda " + str(lambda_a) + " f1 " + str(f1_score))
        if f1_score >= maximum:
            maximum = f1_score
            lambda_a_max = lambda_a
    return lambda_a_max
