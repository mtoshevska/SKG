import numpy as np
import networkx as nx
from argparse import ArgumentParser
from create_edges import load_edges
from create_nodes import load_word_nodes
from create_graph import create_nx_graph


def calculate_node_statistics(nodes):
    """
    Calculates statistics for the given nodes.
    :param nodes: nodes
    :type nodes: pd.DataFrame
    :return: number of nodes, average S1 saliency, average S2 saliency
    :rtype: int, float, float
    """

    total = len(nodes)
    avg_s1_saliency = nodes['S1 Saliency'].values.mean()
    avg_s2_saliency = nodes['S2 Saliency'].values.mean()

    return total, avg_s1_saliency, avg_s2_saliency


def calculate_edge_statistics(edges):
    """
    Calculates statistics for the given edges.
    :param edges: edges
    :type edges: pd.DataFrame
    :return: number of edges, average PMI
    :rtype: int, float
    """

    total = len(edges)
    avg_pmi = edges['PMI'].values.mean() if 'PMI' in edges.columns else 0

    return total, avg_pmi


def calculate_degree_statistics(degrees_list):
    """
    Calculates degree distribution statistics for the given values.
    :param degrees_list: degree distribution list
    :type degrees_list: list
    :return: min value, max value, average value
    :rtype: int, int, float
    """

    degrees_list = [d[1] for d in degrees_list]

    min_value = np.min(degrees_list)
    max_value = np.max(degrees_list)
    avg_value = np.mean(degrees_list)

    return min_value, max_value, avg_value


def calculate_graph_statistics(graph):
    """
    Calculates statistics for the given graph.
    :param degrees_list: graph
    :type degrees_list: nx.MultiGraph
    :return: num nodes, num edges, num connected components, degree distribution list
    :rtype: int, int, int, list
    """

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_cc = nx.number_connected_components(graph)
    node_degrees = list(graph.degree())

    return num_nodes, num_edges, num_cc, node_degrees


def node_statistics(dataset_name, top=False):
    """
    Prints statistics for nodes for the given dataset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param top: whether to include only the 20% most discriminative words or all
    :type top: bool
    """

    categories = ['NOUN', 'VERB', 'ADJ', 'ADV']
    nodes_info = 'Most discriminative 20%' if top else 'All'

    word_nodes = load_word_nodes(dataset_name=dataset_name, top=top)
    total, avg_s1_saliency, avg_s2_saliency = calculate_node_statistics(word_nodes)

    print(f'--- {nodes_info} word nodes\n'
          f'Num nodes: {total}\n'
          f'Average S1 Saliency: {round(avg_s1_saliency, 4)}\n'
          f'Average S2 Saliency: {round(avg_s2_saliency, 4)}\n')

    for category in categories:
        word_nodes_cat = word_nodes[word_nodes['Category'] == category]
        total, avg_s1_saliency, avg_s2_saliency = calculate_node_statistics(word_nodes_cat)

        print(f'----- {nodes_info} word nodes ({category})\n'
              f'Num nodes: {total}\n'
              f'Average S1 Saliency: {round(avg_s1_saliency, 4)}\n'
              f'Average S2 Saliency: {round(avg_s2_saliency, 4)}\n')


def edge_statistics(dataset_name, edge_type, use_existing_word_nodes_only=False, top=False):
    """
    Prints statistics for edges for the given dataset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param edge_type: type of the edge
    :type edge_type: str
    :param use_existing_word_nodes_only: whether to return only the edges
    for word nodes extracted from the style dataset
    :type: bool
    :param top: whether to include only the 20% most discriminative words or all
    :type top: bool
    """

    word_nodes = load_word_nodes(dataset_name=dataset_name, top=top) if use_existing_word_nodes_only else None
    edges = load_edges(dataset_name=dataset_name, edge_type=edge_type, words=word_nodes)
    total, avg_pmi = calculate_edge_statistics(edges)

    print(f'--- {edge_type} edges\n'
          f'Num edges: {total}')
    if 'pmi' in edge_type:
        print(f'Average PMI: {round(avg_pmi, 4)}\n')


def graph_statistics(dataset_name, style_1_name, style_2_name):
    """
    Prints statistics for graph created from the given dataset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param style_1_name: style 1
    :type style_1_name: str
    :param style_2_name: style 2
    :type style_2_name: str
    """

    skg = create_nx_graph(dataset_name=dataset_name, style_1_name=style_1_name, style_2_name=style_2_name)

    num_nodes, num_edges, num_cc, node_degrees = calculate_graph_statistics(skg)

    print(f'--- SKG\n'
          f'Num nodes: {num_nodes}\n'
          f'Num edges: {num_edges}\n'
          f'Num connected components: {num_cc}\n')

    min_degree, max_degree, avg_degree = calculate_degree_statistics(node_degrees)

    print(f'----- Node degree\n'
          f'Min degree: {min_degree}\n'
          f'Max degree: {max_degree}\n'
          f'Average degree: {round(avg_degree, 4)}\n')

    categories = ['NOUN', 'VERB', 'ADJ', 'ADV']
    for category in categories:
        node_degrees_cat = [d for d in node_degrees if category in d[0]]

        min_degree, max_degree, avg_degree = calculate_degree_statistics(node_degrees_cat)

        print(f'------- Node degree ({category})\n'
              f'Min degree: {min_degree}\n'
              f'Max degree: {max_degree}\n'
              f'Average degree: {round(avg_degree, 4)}\n')


if __name__ == '__main__':
    parser = ArgumentParser('Style KG Arguments')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--style_1_name', type=str)
    parser.add_argument('--style_2_name', type=str)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    style_1_name = args.style_1_name
    style_2_name = args.style_2_name

    print('---------------------------------------------\n')
    print(f'Dataset name: {dataset_name}\n')

    node_statistics(dataset_name=dataset_name, top=True)
    print('---------------------------------------------\n')

    edge_statistics(dataset_name=dataset_name, edge_type='synonyms',
                    use_existing_word_nodes_only=True, top=True)
    print('---------------------------------------------\n')
    edge_statistics(dataset_name=dataset_name, edge_type='antonyms',
                    use_existing_word_nodes_only=True, top=True)
    print('---------------------------------------------\n')
    edge_statistics(dataset_name=dataset_name, edge_type='hyponyms',
                    use_existing_word_nodes_only=True, top=True)
    print('---------------------------------------------\n')
    edge_statistics(dataset_name=dataset_name, edge_type='hypernyms',
                    use_existing_word_nodes_only=True, top=True)
    print('---------------------------------------------\n')

    edge_statistics(dataset_name=dataset_name, edge_type=f'pmi_{style_1_name}',
                    use_existing_word_nodes_only=True, top=True)
    print('---------------------------------------------\n')
    edge_statistics(dataset_name=dataset_name, edge_type=f'pmi_{style_2_name}',
                    use_existing_word_nodes_only=True, top=True)
    print('---------------------------------------------\n')

    graph_statistics(dataset_name=dataset_name, style_1_name=style_1_name, style_2_name=style_2_name)
    print('---------------------------------------------\n')
