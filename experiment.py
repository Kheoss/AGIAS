""" 
@Hipotesis 
Among the attack graphs, we can determine patterns

@Methodology
-> For every node A, count the number of nodes of each type B such that there exists an edge from A to B.
-> Transform the count into percentage
-> Sort the results so we have the 'strongest' (highest percentage) on top.  

@Dataset
CPTC-2018
CPTC-2017

@Expected
For certain nodes, I expect patterns that hold for more than 50% of the time

@Results
16 out of the 52 graphs have a subgraf g such that |g| > 0

"""

import AGIAS
import os
import pathlib
import shutil
import matplotlib.pyplot as plt
import numpy as np
import csv

#Parser setup
parser = AGIAS.DotParser(AGIAS.SAGEDataLabelParser())
non_zero_path_graphs = 0
total_graphs = 0

class Touple:
    def __init__(self, from_node, to_node, value):
        self._from = from_node
        self._to = to_node
        self._value = value

    def __str__(self):
        return f"{self._from} -> {self._to} : {self._value}"


with open('patterns_markov_chain_2017.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["begin", "end", "popularity"])
    # interpretability = 0
    patterns = {}
    for file in os.listdir('AG_2017_markov_chain'):
        if file.endswith('.dot'):
            total_graphs += 1
            graph = parser.parseFromFile("AG_2017_markov_chain/" + file)
            patterns = graph.analysePatterns(patterns)
            print(f"Attack graph: {total_graphs}")

    for pattern in patterns:
        total = 0
        print(pattern)
        for end in patterns[pattern]:
            total += patterns[pattern][end]
        for end in patterns[pattern]:
            writer.writerow([pattern, end, patterns[pattern][end]/total])
            print(f"{end} -> {patterns[pattern][end]/total}")
        print("____________________________")
    
