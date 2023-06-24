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

"""
Experiment 2
2018:

Total tests:2850
Total number of discrepancies: 97.0, aprox 3%

Manual analysis:
63 fits the expert's opinion and 34 are consideret too subjective to be 


2017:
Total tests:14028
Total number of discrepancies: 913 aprox 6%


Qualitative analysis ( ACCEPTED | QUESTIONABLE ):
CPTC-2017: 7 | 3 ( 0.013 + 0.28 + 0.178 ) AVG: 0.15


"""


import AGIAS
import os
import pathlib
import shutil
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import stats

#Parser setup
parser = AGIAS.DotParser(AGIAS.SAGEDataLabelParser())
non_zero_path_graphs = 0
total_graphs = 0

class Pair:
    def __init__(self, first, second):
        self._first = first
        self._second = second
    def __str__(self):
        return f"Pair({self._first}, {self._second})"

class Touple:
    def __init__(self, from_node, to_node, value):
        self._from = from_node
        self._to = to_node
        self._value = value

    def __str__(self):
        return f"{self._from} -> {self._to} : {self._value}"

patterns = []
#load patterns

# file1 = open('patterns_2017.csv', 'r')
# lines = file1.readlines()
# count = 0
# for line in lines:
#     if count > 0:
#         parts = line.split(',')
#         patterns.append(Touple(parts[0], parts[1], float(parts[2])))
#     count += 1

explainability_baseline = []
explainability = []

with open(f'pairwise_2017.csv', 'r') as file1:
    lines = file1.readlines()
    count = 0
    for line in lines:
        if count > 0:
            parts = line.split(',')
            explainability_baseline.append(Pair(count, float(parts[1])/2))
            explainability.append(Pair(count, float(parts[0])))
            # print(f"read: {float(parts[0])} : {float(parts[1])/2}")
        count += 1

    # Sort the attack graphs
    explainability.sort(key = lambda x : x._second, reverse=True)
    explainability_baseline.sort(key = lambda x : x._second, reverse=True)
    
    # Create rankings
    ranking_our_aux = [x._first for x in explainability]
    ranking_base_aux = [x._first for x in explainability_baseline]

    ranking_our = [0 for _ in explainability]
    ranking_base = [0 for _ in explainability]

    for index, val in enumerate(ranking_our_aux):
        ranking_our[val-1] = index
    for index, val in enumerate(ranking_base_aux):
        ranking_base[val-1] = index

    # Compute the tau-b value:
    res = stats.kendalltau(ranking_our, ranking_base, variant="b")

    print(f"Result: {res}")

    print("Done")

    # Sampling and debugging the discordant pairs
    """
    # start pariwise comparison
    print("Start pairwise comparison")
    discrepancies = 0
    total_tests = 0
    for i in range(len(explainability)):
        for j in range(i, len(explainability)):
            if i == j:
                continue
            total_tests += 1
            if explainability[i]._second > explainability[j]._second and explainability_baseline[i]._second < explainability_baseline[j]._second:
                print(f"Discrepancy found at between graphs:{i} and {j}: \n baseline[{explainability_baseline[i]._second}.{explainability_baseline[j]._second}] \n new[{explainability[i]},{explainability[j]}]")
                discrepancies += 1
                continue
            
            if explainability[i]._second < explainability[j]._second and explainability_baseline[i]._second > explainability_baseline[j]._second:
                print(f"Discrepancy found at between graphs:{i} and {j}: \n baseline[{explainability_baseline[i]._second}.{explainability_baseline[j]._second}] \n new[{explainability[i]},{explainability[j]}]")
                discrepancies += 1
                continue

    concordant = total_tests - discrepancies
    print(f"Total tests:{total_tests}")
    print(f"Total number of discrepancies: {discrepancies}")
    
    print(f"Tau = {(concordant-discrepancies)/(concordant + discrepancies)}")
    """