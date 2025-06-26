import os
import networkx as nx
from argparse import ArgumentParser
from create_nodes import create_word_nodes, create_word_embeddings, load_word_nodes
from create_edges import create_co_occurrence_with_pmi_edge, find_synonyms, \
    find_antonyms, find_hyponyms, find_hypernyms, load_edges


def create_nx_graph(dataset_name, style_1_name, style_2_name):
    """
    Creates and returns a NetworkX MultiGraph object for the style knowledge graph.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param style_1_name: style 1
    :type style_1_name: str
    :param style_2_name: style 2
    :type style_2_name: str
    :return: style knowledge graph
    :rtype nx.MultiGraph
    """

    skg = nx.MultiGraph()

    word_cat_merge = lambda x: f'{x[0]} <__> {x[1]}'

    word_nodes = load_word_nodes(dataset_name=dataset_name, top=True)
    word_nodes['Node'] = word_nodes[['Word', 'Category']].apply(word_cat_merge, axis=1)
    word_nodes.drop_duplicates(subset='Node', inplace=True)
    word_nodes.set_index('Node', inplace=True)

    skg.add_nodes_from(word_nodes.index)
    nx.set_node_attributes(skg, word_nodes.to_dict(orient='index'))

    pmi_1_edges = load_edges(dataset_name=dataset_name, edge_type=f'pmi_{style_1_name}', words=word_nodes)
    pmi_1_edges['N1'] = pmi_1_edges[['W1', 'Cat1']].apply(word_cat_merge, axis=1)
    pmi_1_edges['N2'] = pmi_1_edges[['W2', 'Cat2']].apply(word_cat_merge, axis=1)
    pmi_1_edges['Attrs'] = pmi_1_edges['PMI'].apply(lambda x: {'pmi_1': x, 'edge_type': 'pmi_1_style'})
    skg.add_edges_from(pmi_1_edges[['N1', 'N2', 'Attrs']].values.tolist())

    pmi_2_edges = load_edges(dataset_name=dataset_name, edge_type=f'pmi_{style_2_name}', words=word_nodes)
    pmi_2_edges['N1'] = pmi_2_edges[['W1', 'Cat1']].apply(word_cat_merge, axis=1)
    pmi_2_edges['N2'] = pmi_2_edges[['W2', 'Cat2']].apply(word_cat_merge, axis=1)
    pmi_2_edges['Attrs'] = pmi_2_edges['PMI'].apply(lambda x: {'pmi_2': x, 'edge_type': 'pmi_2_style'})
    skg.add_edges_from(pmi_2_edges[['N1', 'N2', 'Attrs']].values.tolist())

    synonym_edges = load_edges(dataset_name=dataset_name, edge_type='synonyms', words=word_nodes)
    synonym_edges['N1'] = synonym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    synonym_edges['N2'] = synonym_edges[['Synonym', 'Synonym Category']].apply(word_cat_merge, axis=1)
    synonym_edges['Attrs'] = [{'edge_type': 'synonym'} for _ in range(len(synonym_edges))]
    skg.add_edges_from(synonym_edges[['N1', 'N2', 'Attrs']].values.tolist())

    antonym_edges = load_edges(dataset_name=dataset_name, edge_type='antonyms', words=word_nodes)
    antonym_edges['N1'] = antonym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    antonym_edges['N2'] = antonym_edges[['Antonym', 'Antonym Category']].apply(word_cat_merge, axis=1)
    antonym_edges['Attrs'] = [{'edge_type': 'antonym'} for _ in range(len(antonym_edges))]
    skg.add_edges_from(antonym_edges[['N1', 'N2', 'Attrs']].values.tolist())

    hyponym_edges = load_edges(dataset_name=dataset_name, edge_type='hyponyms', words=word_nodes)
    hyponym_edges['N1'] = hyponym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    hyponym_edges['N2'] = hyponym_edges[['Hyponym', 'Hyponym Category']].apply(word_cat_merge, axis=1)
    hyponym_edges['Attrs'] = [{'edge_type': 'hyponym'} for _ in range(len(hyponym_edges))]
    skg.add_edges_from(hyponym_edges[['N1', 'N2', 'Attrs']].values.tolist())

    hypernym_edges = load_edges(dataset_name=dataset_name, edge_type='hypernyms', words=word_nodes)
    hypernym_edges['N1'] = hypernym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    hypernym_edges['N2'] = hypernym_edges[['Hypernym', 'Hypernym Category']].apply(word_cat_merge, axis=1)
    hypernym_edges['Attrs'] = [{'edge_type': 'hypernym'} for _ in range(len(hypernym_edges))]
    skg.add_edges_from(hypernym_edges[['N1', 'N2', 'Attrs']].values.tolist())

    return skg


if __name__ == '__main__':
    parser = ArgumentParser('Style KG Creation Arguments')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--style_1_name', type=str)
    parser.add_argument('--style_2_name', type=str)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    style_1_name = args.style_1_name
    style_2_name = args.style_2_name

    if not os.path.exists(f'../data/skg/{dataset_name}'):
        os.mkdir(f'../data/skg/{dataset_name}')

    print(f'Creating SKG for dataset {dataset_name}...')

    create_word_nodes(dataset_name=dataset_name, style_1_name=style_1_name, style_2_name=style_2_name)
    create_word_embeddings(dataset_name=dataset_name)

    find_synonyms(dataset_name=dataset_name)
    find_antonyms(dataset_name=dataset_name)
    find_hyponyms(dataset_name=dataset_name)
    find_hypernyms(dataset_name=dataset_name)

    create_co_occurrence_with_pmi_edge(dataset_name=dataset_name, style_name=style_1_name)
    create_co_occurrence_with_pmi_edge(dataset_name=dataset_name, style_name=style_2_name)
