from multiplex import *

def textualize_edges(mp: Multiplex) -> list:
    text = []
    for layer in mp:
        text += [f'{e[0]} is associated with {e[1]}' for e in layer['graph'].edges(data=True)]
    return text

def node_attr(mp: Multiplex) -> list:
    text = [f"node_id {i} is {node_label}" for i, node_label in enumerate(mp.nodes)]
    return text

def layer_attr(mp: Multiplex) -> list:
    text = []
    for i, layer in enumerate(mp.layers):
        text.append(f"layer {i} is from {layer['layer_name']}")
    return text

def node_layer_attr(mp: Multiplex) -> list:
    return node_attr(mp) + layer_attr(mp)

def all_attr(mp: Multiplex) -> list:
    return textualize_edges(mp) + node_layer_attr(mp)

load_textualizer = {
    'edges': textualize_edges,
    'node_attr': node_attr,
    'layer_attr': layer_attr,
    'node_layer_attr': node_layer_attr,
    'all': all_attr
}
