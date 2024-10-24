# import necessary packages
import networkx as nx


def convert_to_nx(filename, graph_type = nx.Graph):

    """
    Function to convert a graph from a file to a networkx graph object

    Parameters:
        filename (str): path to the file containing the graph
        graph_type (networkx.Graph): type of graph to be created
    
    Returns:
        graph (networkx.Graph): networkx graph object
    """

    graph = nx.read_edgelist(filename,
                             create_using=graph_type,
                             nodetype=str,
                             data=(('weight', float),))
    return graph

def nx_to_text(G):
    """
    Function to convert a networkx graph object to a list of nodes and edges

    Parameters:
        G (networkx.Graph): networkx graph object
    
    Returns:
        nodes (list): list of nodes
        edges (list): list of edges
    """

    nodes = []
    for x in G.nodes():
        nodes.append(x)

    # collect edges and weights
    edges = []
    for u,v in G.edges():
        edges.append("("+str(u)+","+str(v)+") with weight " + str(G.get_edge_data(u,v)['weight']))

    return nodes, edges