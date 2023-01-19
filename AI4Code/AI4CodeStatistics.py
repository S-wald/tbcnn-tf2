from AI4Code.AI4CodeNode import AI4CodeNode
from AI4Code.AI4CodeTree import AI4CodeTree
import numpy as np


class AI4CodeStatistics:

    @staticmethod
    def avg(lst):
        return sum(lst) / len(lst)

    @staticmethod
    def analyzeAst(ast: AI4CodeTree):
        __num_nodes = 0
        __max_depth = 0
        __leaf_depths = []
        __max_arity = 0
        __aritys = []
        __ast_vocab = []
        __token_vocab = []

        queue = [(ast.get_root_node(), 1)]
        while queue:
            __num_nodes += 1
            node, cur_depth = queue.pop(0)
            if cur_depth > __max_depth:
                __max_depth = cur_depth
            node_type_rule_name = node.get_type_rule_name()
            __ast_vocab.append(node_type_rule_name)
            node_label = node.get_label()
            if node_label:
                __token_vocab.append(node_label)
            children = node.get_children()
            if not children:
                __leaf_depths.append(cur_depth)
            else:
                arity = len(children)
                __aritys.append(arity)
                if arity > __max_arity:
                    __max_arity = arity
                queue.extend([(child, cur_depth+1) for child in children])
        __avg_leaf_depth = AI4CodeStatistics.avg(__leaf_depths)
        __avg_arity = AI4CodeStatistics.avg(__aritys)
        __median_leaf_depth = np.median(__leaf_depths)
        __median_arity = np.median(__aritys)
        return __num_nodes, __max_depth, __avg_leaf_depth, __max_arity, __avg_arity, __ast_vocab, __token_vocab, __median_leaf_depth, __median_arity
    
    
