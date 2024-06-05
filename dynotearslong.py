from dynotearstrain import train
import os

with open("long.txt", "r") as file:
    lines = file.readlines()
files = ["long" + os.sep + line.strip() for line in lines]

train(files, "long_graph.txt", "long.txt")
