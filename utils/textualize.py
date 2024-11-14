from .multiplex import *

def textualize_edges(mp: Multiplex) -> list[str]:
    text = []
    for layer in mp:
        text += [f'{e[0]} is associated with {e[1]}' for e in layer['graph'].edges(data=True)]
    return text
      

load_textualizer = {
    'edges': textualize_edges,
}
