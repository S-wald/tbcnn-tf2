import json

from AI4Code.AI4CodeEdge import AI4CodeEdge
from AI4Code.AI4CodeNode import AI4CodeNode


class AI4CodeTree:

    def __init__(self, type, order, nodes, edges, number_of_nodes, src_file):
        self.__type = type
        self.__order = order
        self.__nodes = nodes
        self.__edges = edges
        self.__number_of_nodes = number_of_nodes
        self.__src_file = src_file

    def get_nodes(self):
        return self.__nodes

    def get_root_node(self):
        return self.__nodes[0]

    def get_num_of_nodes(self):
        return self.__number_of_nodes

    def get_src_file(self):
        return self.__src_file

    @staticmethod
    def from_json(json_dict):
        ai4code_edges = []
        edges = json_dict['edges']
        for edge in edges:
            ai4code_edge = AI4CodeEdge.from_json(edge)
            ai4code_edges.append(ai4code_edge)

        ai4code_nodes = []
        nodes = json_dict['nodes']
        for node in nodes:
            ai4code_nodes.append(AI4CodeNode.from_json(node))

        for edge in ai4code_edges:
            ai4code_nodes[edge.get_from() - 1].get_children().append(ai4code_nodes[edge.get_to() - 1])

        graph_type = 'tree'
        if 'type' in json_dict:
            graph_type = json_dict['type']
        order = 'bfs'
        if 'order' in json_dict:
            order = json_dict['order']
        num_of_nodes = 2
        if 'num-of-nodes' in json_dict:
            num_of_nodes = json_dict['num-of-nodes']
        src_file = ''
        if 'src-file' in json_dict:
            src_file = json_dict['src-file']
        return AI4CodeTree(graph_type, order, ai4code_nodes, ai4code_edges, num_of_nodes, src_file)
