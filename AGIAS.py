
"""
    AGIAS.py: Attack_Graph_Interpretability_Analysis_and_Simplification 
        
        Quick tool to analyse attack graphs's interpretability.
        Generate alternatives maximising interpretability while minimising the context loss.
        
"""
__author__      = "Vlad-Mihai Constantinescu"

import networkx as nx
import pydot
import re

""" Interface to parse data labels """
class DataLabelParserInterface:
    def parse_data(self, label:str) -> dict:
        """Parse the label data into a dictionary of properities"""
        pass

""" Class to parse the labels of AG's computed with SAGE tool """
class SAGEDataLabelParser(DataLabelParserInterface):
      def parse_data(self, label:str) -> dict:
        #For now we only care about the gap property: 
        return {'gap': int(re.split('gap:|sec', label)[1])}
        

class Node:
    def __init__(self, id, label, alert_signatures, attack_type):
        self._id = id
        self._label = label
        self._alert_signatures = alert_signatures
        self._outgoing_edges = []
        self._ingoing_edges = []
        self._is_sink = False
        self._type = attack_type

    def set_is_sink(self, isSink):
        self._is_sink = isSink

    def add_outgoing_edge(self, edge):
        self._outgoing_edges.append(edge)

    def add_ingoing_edge(self, edge):
        self._ingoing_edges.append(edge)

    def __str__(self):
        return f"Node(id: {self._id}, \
        label: {self._label}, \
        distinct alert signatures: {len(self._alert_signatures)})" 

class Edge:
    def __init__(self, from_node, to_node, label:dict, team):
        self._from = from_node
        self._to = to_node
        self._label = label
        self._team = team                                               # color of the team

    
    def __str__(self):
        return f"Edge(from:{self._from._label}, \
        to:{self._to._label}, \
        label data:{self._label})" 


class Graph:
    def __init__(self, nodes, source_states, sink_states, teams):
        self._nodes = nodes
        self._source_states = source_states
        self._sink_states = sink_states
        self._teams = teams


    def exportAsDot(self):
        graph = pydot.Dot(graph_type='digraph')
        for node in self._nodes:
            for edge in node._outgoing_edges:
                graph.add_edge(pydot.Edge(edge._from._label, edge._to._label, label=str(edge._label), color=edge._team))
                
        graph.write_png("out.png")

    def analysePatterns(self, oldResults):
        results = oldResults

        for node in self._nodes:
            for edge in node._outgoing_edges:
                if node._type in results:
                    if edge._to._type in results[node._type]:
                        results[node._type][edge._to._type] += 1
                    else:
                        results[node._type][edge._to._type] = 1
                else:
                    results[node._type] = {edge._to._type: 1}
        return results
    
    def __str__(self):
        return f"Graph(nr nodes:{len(self._nodes)}, \
        nr source states:{len(self._source_states)}, \
        nr sink states:{len(self._sink_states)})" 

class DotParser:
    def __init__(self, labelDataParser:DataLabelParserInterface):
        self._labelDataParser = labelDataParser

    def parseFromFile(self, path) -> Graph:
        """Parse a .dot file"""
        graph = nx.nx_pydot.read_dot(path)
        node_labels = nx.get_node_attributes(graph, 'tooltip')
        teams = []

        # TO DO :we miss the protocol and targets
        node_map = {label:Node(index,label,node_labels[label].split('\n'), label.split("\\")[0].split("\n")[0]) for index,label in enumerate(node_labels)}
        current_node_index = len(node_labels)
        for node in graph.nodes:                                          # Add nodes that are missing labels
            if node in node_map:
                continue
            node_map[node] = Node(current_node_index, node, [], node.split("\n")[1])
            current_node_index += 1

        # check for duplicates
        edgeExistance = [[False for _ in range(len(node_map))] for _ in range(len(node_map))]

        for e in graph.edges:
            if(edgeExistance[node_map[e[0]]._id][node_map[e[1]]._id]):
                continue

            edgeExistance[node_map[e[0]]._id][node_map[e[1]]._id] = True
            for _x in graph[e[0]][e[1]]:
                ed = graph[e[0]][e[1]][_x]
                edge = None
                team = "black"
                if 'color' in ed:                                           # different colors determine different attacker teams
                    team = ed['color']
                if team != "black" and team not in teams:
                    teams.append(team)
                if 'label' in ed:
                    labelData = self._labelDataParser.parse_data(ed['label'])
                    edge = Edge(node_map[e[0]], node_map[e[1]], labelData, team)
                else:
                    node_map[e[1]].set_is_sink(True)                        # No edge data to him, so it is a sink
                    edge = Edge(node_map[e[0]], node_map[e[1]], {}, team)
                    
                node_map[e[0]].add_outgoing_edge(edge)
                node_map[e[1]].add_ingoing_edge(edge)
                
        nodes = [node_map[_x] for _x in node_map]
        source_states = [node_map[_x] for _x in node_map if len(node_map[_x]._ingoing_edges)==0]
        sink_states = [node_map[_x] for _x in node_map if len(node_map[_x]._outgoing_edges)==0]

        return Graph(nodes, source_states, sink_states, teams)
#MAIN

# parser = DotParser(SAGEDataLabelParser())
# graph = parser.parseFromFile("AG_2018/ag1.dot")

# patterns = []
# patterns = graph.analysePatterns(patterns)
# graph.exportAsDot()
# # print(patterns)
# for pattern in patterns:
#     print(pattern)
#     print(patterns[pattern])
#     print("________")