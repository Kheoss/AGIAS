""" 
@Results
SIZE = 10 nodes : 
    Planarity of density of 10.0%
    1.0
    Planarity of density of 20.0%
    0.8
    Planarity of density of 30.000000000000004%
    0.3073809523809525
    Planarity of density of 40.0%
    0.09274136974136975
    Planarity of density of 50.0%
    0.049452143660321066
    Planarity of density of 60.0%
    0.03431695141194749
    Planarity of density of 70.0%
    0.02708771566071062
    Planarity of density of 80.0%
    0.02114420277161496
    Planarity of density of 89.99999999999999%
    0.017587015729356597
    

SIZE = 20 nodes:
Planarity of density of 10.0%
0.23936507936507936
Planarity of density of 20.0%
0.02347096085881809
Planarity of density of 30.000000000000004%
0.012362843648924895
Planarity of density of 40.0%
0.008498551782834382
Planarity of density of 50.0%
0.006411368630411492
Planarity of density of 60.0%
0.0051688800659916035
Planarity of density of 70.0%
0.0031
Planarity of density of 80.0%
0.0024
Planarity of density of 90.0%
0.0012


SIZE = 30 nodes:
Planarity of density of 10.0%
0.021027355121967285
Planarity of density of 20.0%
0.007342761646545383
Planarity of density of 30.000000000000004%
0.004636765177049606

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
file1 = open('patterns_2017.csv', 'r')
lines = file1.readlines()
count = 0
for line in lines:
    if count > 0:
        parts = line.split(',')
        patterns.append(Touple(parts[0], parts[1], float(parts[2])))
    count += 1


with open(f'pairwise.csv', 'w', newline='') as csvfile:    
    explainability_baseline = []
    explainability = []
    for file in os.listdir('AG_2017/ExperimentAGs'):
        if file.endswith('.dot'):
            total_graphs += 1
            graph = parser.parseFromFile("AG_2017/ExperimentAGs/" + file)
            chunks = graph.splitIntoCognitiveChunks(patterns)
            interdependency = graph.calculateInterdependencyPerChunkWithClosenessCentrality(chunks)
            radial_spread = graph.calculateRadialSpread()
            monotonicity = graph.calculateMonotonicity()
            chunks_number = len(set(chunks))
            
            
            print(f"_______________________________Graph:{total_graphs}_______________________________")
            planarity, relative_planarity = graph.calculatePlanarity()

            # calculate explainability 
            base_explainability = (1/chunks_number + (1-interdependency))/2
            my_explainability = (1/chunks_number + (1-interdependency) + planarity)/3
            
            explainability_baseline.append(base_explainability)
            explainability.append(my_explainability)

            graph.exportAsDot(str(total_graphs))
            graph.exportAsDotPerChunks(str(total_graphs), chunks)

    # start pariwise comparison
    print("Start pairwise comparison")
    discrepancies = 0
    for i in range(len(explainability)):
        for j in range(len(explainability)):
            if i == j:
                continue
            if explainability[i] > explainability[j] and my_explainability[i] < my_explainability[j]:
                print(f"Discrepancy found at between graphs:{i} and {j}")
                discrepancies += 1
            
            if explainability[i] < explainability[j] and my_explainability[i] > my_explainability[j]:
                print(f"Discrepancy found at between graphs:{i} and {j}")
                discrepancies += 1


    print(f"Total number of discrepancies: {discrepancies}")