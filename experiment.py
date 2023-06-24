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

patterns = []
#load patterns
file1 = open('patterns_2018.csv', 'r')
lines = file1.readlines()
count = 0
for line in lines:
    if count > 0:
        parts = line.split(',')
        patterns.append(Touple(parts[0], parts[1], float(parts[2])))
    count += 1

with open('PatternsWithBetweennessResults_2018.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(["GraphNo", "ChunksNo", "Interdependency", "Explainability"])
    interpretability = 0 
    for file in os.listdir('AG_2017_markov_chain'):
        if file.endswith('.dot'):
            total_graphs += 1
            graph = parser.parseFromFile("AG_2017_markov_chain/" + file)
    
            chunks = graph.splitIntoCognitiveChunks(patterns)
            # graph.exportAsDotPerChunks(str(total_graphs), chunks)
            # break;        
            # print(chunks)
            # interd = graph.calculateInterdependencyPerChunk(chunks)
            interdependency = graph.calculateInterdependencyPerChunk(chunks)
            # break
            # calculate explainability 
            chunks_number = len(set(chunks))            
            print(f"_______________________________Graph:{total_graphs}_______________________________")
            print(f"Number of chunks: {chunks_number}")
            print(f"Interdependency: {interdependency}")
            
            planarity, relative_planarity = graph.calculatePlanarity()
            explainability = (1/chunks_number + (1-interdependency) + planarity)/3

            print(f"Planarity: {planarity}")
            print(f"Explainability: {explainability}")
            
            interpretability += explainability
            # writer.writerow([total_graphs, chunks_number, interdependency, explainability])
            graph.exportAsDot(str(total_graphs))
            graph.exportAsDotPerChunks(str(total_graphs), chunks)
            # break
    print(f"Average interpretability : {interpretability/total_graphs}")
# TO DO :
# [ ] Find multiple ways to calculate interdependency
# [ ] Find multiple ways of chuncking
# 