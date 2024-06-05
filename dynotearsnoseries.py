from dynotearstrain import train

with open("series.txt", "r") as file:
    lines = file.readlines()
files = [line.strip() for line in lines]

train(files, "series_graph.txt", "results.txt")
