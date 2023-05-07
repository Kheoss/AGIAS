
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
    def __init__(self, id, label, alert_signatures):
        self._id = id
        self._label = label
        self._alert_signatures = alert_signatures
        self._outgoing_edges = []
        self._ingoing_edges = []
        self._is_sink = False

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
    def __init__(self, id, from_node, to_node, label:dict):
        self._id = id
        self._from = from_node
        self._to = to_node
        self._label = label

    
    def __str__(self):
        return f"Edge(from:{self._from._label}, \
        to:{self._to._label}, \
        label data:{self._label})" 


class Graph:
    def __init__(self, nodes, source_states, sink_states):
        self._nodes = nodes
        self._source_states = source_states
        self._sink_states = sink_states

    def exportAsDot(self, file_name):
        graph = pydot.Dot(graph_type='digraph')
        for node in self._nodes:
            for edge in node._outgoing_edges:
                graph.add_edge(pydot.Edge(edge._from._label, edge._to._label, label=str(edge._label)))
                
        graph.write_png(file_name+".png")

    def computeNumberOfPathsFromSourcesToSink(self) -> int:
        self.computeNumberOfPathsFromSourcesToSinkIgnore([])

    def computeNumberOfPathsFromSourcesToSinkIgnore(self, nodes_to_ignore):
        nr_paths = 0
        for source in self._source_states:
            #compute nr of paths from source to any sink
            nr_paths += self.computeNumberOfPathsFromSource(source, nodes_to_ignore, []) 

        return nr_paths
    # DFS
    def computeNumberOfPathsFromSource(self, current, nodes_to_ignore, visited) -> int:
        if current._id in nodes_to_ignore:
            return 0
        if current._id in [sink._id for sink in self._sink_states]:
            return 1
        paths = 0
        for edge in current._outgoing_edges:
            if edge._id in visited:
                continue
            paths += self.computeNumberOfPathsFromSource(edge._to, nodes_to_ignore, visited + [edge._id])

        return paths

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

        node_map = {label:Node(index,label,node_labels[label].split('\n')) for index,label in enumerate(node_labels)}
        current_node_index = len(node_labels)
        for node in graph.nodes:                                          # Add nodes that are missing labels
            if node in node_map:
                continue
            node_map[node] = Node(current_node_index, node, []);
            current_node_index += 1

        # check for duplicates
        edgeExistance = [[False for _ in range(len(node_map))] for _ in range(len(node_map))]
        edgeIndex = 0
        for e in graph.edges:
            if(edgeExistance[node_map[e[0]]._id][node_map[e[1]]._id]):
                continue

            edgeExistance[node_map[e[0]]._id][node_map[e[1]]._id] = True

            for _x in graph[e[0]][e[1]]:
                ed = graph[e[0]][e[1]][_x]
                edge = None
                if 'label' in ed:
                    labelData = self._labelDataParser.parse_data(ed['label'])
                    edge = Edge(edgeIndex, node_map[e[0]], node_map[e[1]], labelData)
                else:
                    node_map[e[1]].set_is_sink(True)                        # No edge data to him, so it is a sink
                    edge = Edge(edgeIndex, node_map[e[0]], node_map[e[1]], {})
                    
                node_map[e[0]].add_outgoing_edge(edge)
                node_map[e[1]].add_ingoing_edge(edge)
                # print(edgeIndex)
                edgeIndex += 1
                
        nodes = [node_map[_x] for _x in node_map]
        source_states = [node_map[_x] for _x in node_map if len(node_map[_x]._ingoing_edges)==0]
        sink_states = [node_map[_x] for _x in node_map if len(node_map[_x]._outgoing_edges)==0]
        
        return Graph(nodes, source_states, sink_states)

# MAIN

# parser = DotParser(SAGEDataLabelParser())
# graph = parser.parseFromFile("AG/ag1.dot")

# graph.exportAsDot()
