import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import read_graph
from utils import path


def f1_series(results_file, target_file):
    dynotears = pd.read_table(results_file)
    plt.figure()
    plt.plot(dynotears['Nseries'], dynotears['F1'], marker='o', linestyle='-', label="DyNoTears")
    plt.xlabel('Number of series')
    plt.ylabel('F1-score')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(target_file)


def f1lo_series(results_file, target_file):
    dynotears = pd.read_table(results_file)
    plt.figure()
    plt.errorbar(dynotears['Nseries'], dynotears['F1LO'], yerr=dynotears['F1LOstd'], marker='o', linestyle='-', ecolor='black', capsize=5, label="DyNoTears")
    plt.xlabel('Number of series')
    plt.ylabel('F1-score-CV')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(target_file)


def time_series(results_file, target_file):
    dynotears = pd.read_table(results_file)
    plt.figure()
    #plt.plot(dynotears['Nseries'], dynotears['Time'], marker='o', linestyle='-')
    plt.errorbar(dynotears['Nseries'], dynotears['Time'], yerr=dynotears['Timestd'], marker='o', linestyle='-', ecolor='black', capsize=5, label="DyNoTears")
    plt.xlabel('Number of series')
    plt.ylabel('Time (sec)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(target_file)


def plot_graph(graph_file, output_file):
    graph = read_graph(graph_file)
    adjacency = np.zeros((16, 16))

    labels = ["FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE", "OXALOACETATE", "FAD", "SUCCINYL-COA", "NAD",
              "A-K-GLUTARATE", "GDP", "NADH", "CITRATE", "SUCCINATE", "ISOCITRATE", "ACETY-COA"]

    for u, v in graph:
        uind = labels.index(u[:-5]) # strip the _lag
        vind = labels.index(v[:-5])
        adjacency[uind, vind] = 1

    plt.figure()
    plt.imshow(adjacency, cmap='binary', interpolation='nearest', vmin=-1, vmax=1)
    plt.xticks(ticks=np.arange(len(adjacency)), labels=labels, rotation=80)
    plt.yticks(ticks=np.arange(len(adjacency)), labels=labels)
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(output_file)


f1_series("results.txt", 'series_f1.pdf')
f1lo_series("results.txt", 'series_f1lo.pdf')
time_series("results.txt", 'series_time.pdf')
f1_series("threes.txt", 'threes_f1.pdf')
f1lo_series("threes.txt", 'threes_f1lo.pdf')
time_series("threes.txt", 'threes_time.pdf')

plot_graph("series_graph.txt", "series_dynotears.pdf")
plot_graph("threes_graph.txt", "threes_dynotears.pdf")
plot_graph(path + os.sep + "groundtruth.txt", "groundtruth.pdf")
