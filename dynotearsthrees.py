from dynotearstrain import train

files = ["threes"+str(i)+".tsv" for i in range(100)]

train(files, graph_file="threes_graph.txt", results_file="threes.txt")
