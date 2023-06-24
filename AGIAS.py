
"""
    AGIAS.py: Attack_Graph_Interpretability_Analysis_and_Simplification 
        
        Quick tool to analyse attack graphs's interpretability.
        Generate alternatives maximising interpretability while minimising the context loss.
        
"""
__author__      = "Vlad-Mihai Constantinescu"

from itertools import combinations
import networkx as nx
import pydot
import re
import math
import random

class GraphUtil:
    @staticmethod
    def generateGraphWithMisses(size, missing_edges):
        nodes = [ Node(i,i,[],"") for i in range(size) ]

        edge_count = -1
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                edge_count+=1
                if edge_count in missing_edges:
                    continue

                nodes[i].add_outgoing_edge(Edge(nodes[i], nodes[j], {}, "black"))
        return Graph(nodes, [], [], [])

    @staticmethod
    def generateRandomGraph(size, density):
        max_links = size * (size-1)
        required_links = density * max_links
        links_to_be_removed = max_links - required_links
        # generate a list of random edges
        edges_to_remove = []
        while len(edges_to_remove) < links_to_be_removed:
            new_edge = random.randint(0, max_links-1)
            if new_edge in edges_to_remove:
                continue
            edges_to_remove.append(new_edge)

        # generate the graph
        return GraphUtil.generateGraphWithMisses(size, edges_to_remove)        
        


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

class GraphUtility:
    @staticmethod
    def isPlanarWithoutEdges(graph, edges):
        nx_graph = graph.transformToNXGraph(simplify=True, miss=edges)
        return nx.is_planar(nx_graph)

# make the utility methods static
# GraphUtility.isPlanarWithoutEdges = staticmethod(GraphUtility.isPlanarWithoutEdges)

class Graph:
    def __init__(self, nodes, source_states, sink_states, teams):
        self._nodes = nodes
        self._source_states = source_states
        self._sink_states = sink_states
        self._teams = teams
        self._chunks = []
        self.graph_paths = []


    def exportAsDot(self, name):
        graph = pydot.Dot(graph_type='digraph')
        for node in self._nodes:
            for edge in node._outgoing_edges:
                graph.add_edge(pydot.Edge(edge._from._label, edge._to._label, label=str(edge._label), color=edge._team))
                
        graph.write_png(f"ResultedAGs/{name}.png")

    def exportAsDotPerChunks(self, name, chunks):
        # colors 
        colors = ["peru", "navyblue", "olive", "olivedrab1",
        "turquoise","orangered1","orchid1","palegreen","paleturquoise","palevioletred","papayawhip","pink"
        ,"plum3","purple2","salmon","seagreen4","tan2","violet","turquoise","webpurple","yellow","plum","gray1","fuchsia","dodgerblue2"]

        # assign colors to each chunk
        color_index = 0
        chunk_color_assignment = dict()
        for chunk in chunks:
            if chunk_color_assignment.get(chunk) is None:
                chunk_color_assignment.update({chunk: colors[color_index]})
                color_index += 1

        graph = pydot.Dot(graph_type='digraph')
        for node in self._nodes:
            node_color = chunk_color_assignment.get(chunks[node._id])
            # if len([n for n in self._sink_states if n._id == node._id]) > 0:
            #     node_color = "white"
            graph.add_node(pydot.Node(node._id, style='filled', fillcolor=node_color, label=node._label))
        for node in self._nodes:
            for edge in node._outgoing_edges:
                graph.add_edge(pydot.Edge(edge._from._id, edge._to._id, label=str(edge._label), color=edge._team))
                
        graph.write_png(f"ResultedAGs/{name}_chunks.png")

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
    
    def splitIntoCognitiveChunks(self, patterns):
        # sort patterns
        patterns.sort(key=lambda t: t._value, reverse=True)

        node_chunk = [node._id for node in self._nodes]
        node_assigned = [False for _ in range(len(self._nodes))]

        for pattern in patterns:
            # search for pattern
            for node in self._nodes:
                if node._type == pattern._from:
                    # search for end of pattern
                    for edge in node._outgoing_edges:
                        if edge._to._type == pattern._to and node_assigned[edge._to._id] == False:
                            node_assigned[edge._to._id] = True
                            chunk_to_merge = node_chunk[edge._to._id]
                            
                            # merge all chunk to the smaller chunk
                            # TO DO: Change the algorithm to also implement path compression? Investigate
                            for ind in range(len(node_chunk)):
                                if node_chunk[ind] == chunk_to_merge:
                                    node_chunk[ind] = node_chunk[node._id]
                            break
        self._chunks = node_chunk
        return node_chunk
 
    # create graph only with nodes inside one chunk         
    def subgraphByChunk(self, chunk_allocation, chunk):
        new_nodes = {}
        for node in self._nodes:
            if chunk_allocation[node._id] == chunk:
                new_nodes.update({node._id: Node(id=node._id, label=node._label, alert_signatures=node._alert_signatures, attack_type=node._type)})

        for node in self._nodes:
            if chunk_allocation[node._id] == chunk:
                for edge in node._outgoing_edges:
                    if chunk_allocation[edge._to._id] == chunk:
                        # add edge from node to edge._to
                        e = Edge(new_nodes.get(node._id), new_nodes.get(edge._to._id), edge._label, edge._team)
                        new_nodes.get(node._id).add_outgoing_edge(e)  
                    # def __init__(self, id, label, alert_signatures, attack_type):
                    # def __init__(self, from_node, to_node, label:dict, team):
                    # def __init__(self, nodes, source_states, sink_states, teams):

        graph_nodes = [new_nodes.get(key) for key in new_nodes]
        return Graph(graph_nodes, [], [], self._teams)

    def _edgeInList(self, target, list):
        for e in list:
            if e._to._id == target._to._id and e._from._id == target._from._id:
                return True

        return False

    # put the graph into an nx friendly form
    def transformToNXGraph(self, simplify=False, miss=[]):
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from([n._id for n in self._nodes])

        max_graph_id = 0
        for node in self._nodes:
            if node._id > max_graph_id:
                max_graph_id = node._id
            for edge in node._outgoing_edges:
                if edge._to._id > max_graph_id:
                    max_graph_id = edge._to._id

        max_graph_id += 1


        connection_matrix = [[False for _ in range(max_graph_id)] for _ in range(max_graph_id)]
        
        for node in self._nodes:
            for edge in node._outgoing_edges:
                if simplify and connection_matrix[node._id][edge._to._id]:
                    continue
                # skip the missing edge
                if self._edgeInList(edge, miss):
                    continue

                nx_graph.add_edge(node._id, edge._to._id)
                # print(node._id)
                # print(edge._to._id)
                # print(max_graph_id)
                connection_matrix[node._id][edge._to._id] = True

        return nx_graph

    # put the graph into an nx friendly undirected form
    def transformToNXUndirectedGraph(self):
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from([n._id for n in self._nodes])

        for node in self._nodes:
            for edge in node._outgoing_edges:
                nx_graph.add_edge(node._id, edge._to._id)
                nx_graph.add_edge(edge._to._id, node._id)
        
        return nx_graph
    
    def getSimpleEdges(self):
        result = []
        connection_matrix = [[False for _ in self._nodes] for _ in self._nodes]
        
        for node in self._nodes:
            for edge in node._outgoing_edges:
                if connection_matrix[node._id][edge._to._id]:
                    continue
                
                result.append(edge)
                connection_matrix[node._id][edge._to._id] = True

        return result

    def getAllEdges(self):
        result = []
        
        for node in self._nodes:
            for edge in node._outgoing_edges: 
                result.append(edge)

        return result

    """
        Calculate the minimum number of edges that can be removed so the graph is planar
    """
    def calculatePlanarity(self):
        simpleEdges = self.getSimpleEdges()
        for i in range(len(simpleEdges)):
            # print(f"{i}:{len(simpleEdges)}")
            is_planar = False
            sample_number = 0
            for comb in combinations(simpleEdges, i):
                sample_number += 1
                if sample_number > 1000:
                    break
                if GraphUtility.isPlanarWithoutEdges(self, list(comb)):
                    is_planar = True
                    break

            if is_planar:
                return 1/(i+1), 1- (i)/len(simpleEdges) 

    def calculateLinkDensity(self):
        # sqrt(#links/nodes^2)
        return math.sqrt(len(self.getSimpleEdges()) /len(self._nodes)**2)



    def calculateBetweennessCentrality(self):
        nx_graph = self.transformToNXGraph()

        betweenness = nx.betweenness_centrality(nx_graph)    
        return betweenness

    def calculateClosenessCentrality(self):
        nx_graph = self.transformToNXGraph()
        closeness = nx.closeness_centrality(nx_graph)

        return closeness

    def getNodeById(self, id):
        for n in self._nodes:
            if n._id == id:
                return n


        raise f"Node with ID :{id} not found in node list"

    def calculateMonotonicity(self):
        average_ratio = 0
        node_count = 0
        for node in self._nodes:
            len_out = len(node._outgoing_edges)
            len_in = len(node._ingoing_edges)
            
            _min = min(len_out, len_in)
            _max = max(len_out, len_in)

            if _min == 0:
                continue
            node_count += 1
            average_ratio += _min/_max

        average_ratio /= node_count

        return average_ratio

    def calculateRadialSpread(self):
        closeness = self.calculateClosenessCentrality()
        max_key = None
        max_val = 0
        # get the node with the highest closeness centrality
        central_node_id = max(closeness, key=closeness.get, default=None)
        central_node = self.getNodeById(central_node_id)

        # calculate distances between all nodes and the central nodes
        nx_graph = self.transformToNXUndirectedGraph()
        path_lengths = nx.shortest_path_length(nx_graph, source=central_node_id)
        
        # find the furtherst and closest nodes
        # print(path_lengths)
        print(central_node)
        highest_distance = max(path_lengths.values())
        smallest_distance = min(path_lengths.values())

        # divide the lowest distance to the highest distance and return the oposite
        return 1 - (smallest_distance/highest_distance)

    def calculateInterdependencyPerChunk(self, chunk_allocation, VERBOSE=False):
        chunks = set(chunk_allocation)
        index = 0
        average_interdependency_among_chunks = 0
        for chunk in chunks:
            chunk_graph = self.subgraphByChunk(chunk_allocation, chunk)

            # calculate betweeness centrality
            betweeness = chunk_graph.calculateBetweennessCentrality()

            # find max of betweeness
            max_betweeness = max(betweeness.values())
            # find key nodes
            key_nodes = [n for n in chunk_graph._nodes if betweeness.get(n._id) == max_betweeness]
            if VERBOSE:
                print(f"________________________Chunk:{chunk}_________________________")
                print(f"Number of nodes: {len(chunk_graph._nodes)}")
                print("Key nodes:")
                for n in key_nodes:
                    print(n)

            # 1 node => full interdependency
            if len(chunk_graph._nodes) == 1:
                average_interdependency_among_chunks += 1
                if VERBOSE:
                    print(f"Interdependency: 1")
                continue

            # calculate interdependency as average of interconectivity between the key_nodes and the other nodes in the group\
            teams_in_graph = dict()
            for n in chunk_graph._nodes:
                for e in n._outgoing_edges:
                    teams_in_graph.update({e._team: True})
            average_interdependency = 0
            for n in key_nodes:
                average_interdependency += len(n._outgoing_edges) / ((len(chunk_graph._nodes)-1) * len(teams_in_graph)) 

            average_interdependency /= len(key_nodes)
            average_interdependency_among_chunks += average_interdependency
            if VERBOSE:
                print(f"Interdependency: {average_interdependency}")

        average_interdependency_among_chunks /= len(chunks)
        return average_interdependency_among_chunks
    
    def calculateInterdependencyPerChunkWithClosenessCentrality(self, chunk_allocation, VERBOSE=False):
        chunks = set(chunk_allocation)
        index = 0
        average_interdependency_among_chunks = 0
        for chunk in chunks:
            chunk_graph = self.subgraphByChunk(chunk_allocation, chunk)

            # calculate betweeness centrality
            betweeness = chunk_graph.calculateClosenessCentrality()

            # find max of betweeness
            max_betweeness = max(betweeness.values())
            # find key nodes
            key_nodes = [n for n in chunk_graph._nodes if betweeness.get(n._id) == max_betweeness]
            if VERBOSE:
                print(f"________________________Chunk:{chunk}_________________________")
                print(f"Number of nodes: {len(chunk_graph._nodes)}")
                print("Key nodes:")
                for n in key_nodes:
                    print(n)

            # 1 node => full interdependency
            if len(chunk_graph._nodes) == 1:
                average_interdependency_among_chunks += 1
                if VERBOSE:
                    print(f"Interdependency: 1")
                continue

            # calculate interdependency as average of interconectivity between the key_nodes and the other nodes in the group\
            teams_in_graph = dict()
            for n in chunk_graph._nodes:
                for e in n._outgoing_edges:
                    teams_in_graph.update({e._team: True})
            average_interdependency = 0
            for n in key_nodes:
                average_interdependency += len(n._outgoing_edges) / ((len(chunk_graph._nodes)-1) * len(teams_in_graph)) 

            average_interdependency /= len(key_nodes)
            average_interdependency_among_chunks += average_interdependency
            if VERBOSE:
                print(f"Interdependency: {average_interdependency}")

        average_interdependency_among_chunks /= len(chunks)
        return average_interdependency_among_chunks

    def findPath(self, chunk_allocation, visited, current_node, team):
        visited[current_node._id] = True
        if len([n for n in self._sink_states if n._id == current_node._id]) > 0:
            # analyse how much of this path was on a single chunk allocation
            chunk_count = dict()
            for chunk in set(chunk_allocation):
                chunk_count.update({chunk: 0})

            for node in self._nodes:
                if visited[node._id]:
                    ct = chunk_count.get(chunk_allocation[node._id])
                    chunk_count.update({chunk_allocation[node._id]: ct+1})
            path_length = 0
            for cnk in chunk_count:
                path_length += chunk_count.get(cnk)

            self.graph_paths.append([chunk_count.get(cnk)/path_length for cnk in chunk_count])

            visited[current_node._id] = False
            return    
        for edge in current_node._outgoing_edges:
            if (edge._team != team and not edge._to._is_sink) or visited[edge._to._id]:
                continue
            self.findPath(chunk_allocation, visited, edge._to, team)
            
        visited[current_node._id] = False

    # returns the area ocupied by different chunks in all possible paths
    def analyseChunksInPaths(self, chunk_allocation):
        for team in self._teams:
            for source in self._source_states:
                visited = [False for n in self._nodes]
                self.findPath(chunk_allocation, visited, source, team)
        
        paths_number = 1
        result = [0 for ck in set(chunk_allocation)]
        if len(self.graph_paths) > 0:
            paths_number = len(self.graph_paths)
        for row in range(len(self.graph_paths)):
            for index, val in enumerate(self.graph_paths[row]):
                result[index] += val
        
        for ind in range(len(result)):
            result[ind] /= paths_number
        return result

  

    def __str__(self):
        return f"Graph(nr nodes:{len(self._nodes)}, \
        nr source states:{len(self._source_states)}, \
        nr sink states:{len(self._sink_states)})" 

class DotParser:
    def __init__(self, labelDataParser:DataLabelParserInterface):
        self._labelDataParser = labelDataParser

    def allEdgesLoop(self, node):
        if len(node._ingoing_edges) == 0:
            return True

        # check if all ingoing edges are loop edges
        for edge in node._ingoing_edges:
            if edge._from._id != node._id:
                return False

        return True

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
        source_states = [node_map[_x] for _x in node_map if self.allEdgesLoop(node_map[_x]) ]
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