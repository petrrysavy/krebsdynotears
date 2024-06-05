from dynotearstrain import train
import os

with open("short.txt", "r") as file:
    lines = file.readlines()
files = ["short" + os.sep + line.strip() for line in lines]

train(files, "short_graph.txt", "short.txt", measurements=range(1, 10000, 100))
