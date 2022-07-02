import graphviz


def visualize_ast(ast):
    dot = graphviz.Digraph('ast')
    queue = [(ast, -1)]
    node_id = 0
    while queue:
        node, parent_node_id = queue.pop(0)
        node_type_rule_name = node.get_type_rule_name()
        dot_node_label = node_type_rule_name
        node_label = node.get_label()
        if node_label:
            dot_node_label += " |" + node_label + "|"
        dot.node(str(node_id), dot_node_label)
        queue.extend([(child, node_id) for child in node.get_children()])
        if parent_node_id > -1:
            dot.edge(str(parent_node_id), str(node_id), dir='none')
        node_id += 1
    return dot
