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


SAMPLE_SIZE = 10
GRAPH_SIZE = 30
EDGE_DENSITY = 0.25

with open(f'Generator_{GRAPH_SIZE}.csv', 'w', newline='') as csvfile:
    # writer = csv.writer(csvfile)
    # writer.writerow(["GraphNo", "Size", "Link Density 25%", "Planarity 25%", "Link Density 50%", "Planarity 50%", "Link Density 75%", "Planarity 75%"])


    density = 0.1
    while density < 1:
        sum_of_planarity = 0
        print(f"Planarity of density of {density*100}%")
        for i in range(SAMPLE_SIZE):
            graph = AGIAS.GraphUtil.generateRandomGraph(GRAPH_SIZE, density) 
            # link_density_2 = graph.calculateLinkDensity()
            planarity_2, relative_planarity_2 = graph.calculatePlanarity()
            sum_of_planarity += planarity_2

        density += 0.1
        avg_planarity = sum_of_planarity/SAMPLE_SIZE
        print(avg_planarity)

        # writer.writerow([i, GRAPH_SIZE, link_density_1, planarity_1, link_density_2, planarity_2, link_density_3, planarity_3])
    
    # for file in os.listdir('AG_2017/ExperimentAGs'):
        # if file.endswith('.dot'):
            # total_graphs += 1
            # graph = parser.parseFromFile("AG_2017/ExperimentAGs/" + file)
            # chunks = graph.splitIntoCognitiveChunks(patterns)
            # interdependency = graph.calculateInterdependencyPerChunkWithClosenessCentrality(chunks)
            # radial_spread = graph.calculateRadialSpread()
            # monotonicity = graph.calculateMonotonicity()
            # calculate explainability 
            # chunks_number = len(set(chunks))
            # explainability = 1/chunks_number + (1-interdependency) + monotonicity
            # explainability = 1/chunks_number + (1-interdependency)
            
            # print(f"_______________________________Graph:{total_graphs}_______________________________")
            # planarity, relative_planarity = graph.calculatePlanarity()
            # link_density = graph.calculateLinkDensity()
            # print(f"Planarity:{planarity}")
            # print(f"Relative-planarity:{relative_planarity}")
            # print(f"Link-density:{link_density}")
            # print(f"Number of chunks: {chunks_number}")
            # print(f"Interdependency: {interdependency}")
            # # print(f"Monotonicity: {monotonicity}")
            # print(f"Explainability: {explainability}")
            # edges_number = 0
            # for n in graph._nodes:
            #     edges_number += len(n._outgoing_edges)

            # writer.writerow([total_graphs, planarity, relative_planarity, link_density])

            # graph.exportAsDot(str(total_graphs))
            # graph.exportAsDotPerChunks(str(total_graphs), chunks)
            # break            
