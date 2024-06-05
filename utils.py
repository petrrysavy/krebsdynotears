import os

path = ("data")

def f1score(predictions, groundtruth):
    union = groundtruth.union(predictions)
    intersection = groundtruth.intersection(predictions)
    tp = len(intersection)
    fp = len(union) - len(groundtruth)
    fn = len(union) - len(predictions)
    return tp / (tp + (fp + fn) / 2)


def read_graph(graph_file):
    with open(graph_file, "r") as file:
        lines = file.readlines()
    groundtruth = [tuple(line.strip().split()) for line in lines]
    return set(groundtruth)


def write_graph(edges_set, graph_file):
    with open(graph_file, "w") as file:
        for u, v in edges_set:
            file.write(u + " " + v + "\n")
