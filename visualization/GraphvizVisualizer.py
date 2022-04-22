import graphviz


class GraphvizVisualizer:
    def visualizeAst(self, ast, node_importances):
        dot = graphviz.Digraph('ast')
        queue = [(ast['tree'], -1)]
        node_id = 0
        while queue:
            node, parent_node_id = queue.pop(0)
            dot.node(str(node_id), node['node'],
                     fillcolor="0.33 {h:} 1".format(h=node_importances[node_id]/len(node_importances)),
                     style='filled')
            queue.extend([(child, node_id) for child in node['children']])
            if parent_node_id > -1:
                dot.edge(str(parent_node_id), str(node_id), dir='none')
            node_id += 1
        return dot




