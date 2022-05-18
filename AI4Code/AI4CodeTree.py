import json

from AI4Code.AI4CodeEdge import AI4CodeEdge
from AI4Code.AI4CodeNode import AI4CodeNode


class AI4CodeTree:

    def __init__(self, type, order, nodes, edges):
        self.__type = type
        self.__order = order
        self.__nodes = nodes
        self.__edges = edges

    def get_nodes(self):
        return self.__nodes

    def get_root_node(self):
        return self.__nodes[0]

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

        return AI4CodeTree(json_dict['type'], json_dict['order'], ai4code_nodes, ai4code_edges)
