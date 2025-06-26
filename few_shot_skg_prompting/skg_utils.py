import pandas as pd
import networkx as nx
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer, WhitespaceTokenizer, word_tokenize

skg_root_location = '../data'

word_cat_merge = lambda x: f'{x[0]} <__> {x[1]}'


def tokenize_and_tag(dataset_name, data):
    """
    Tokenize samples and determine the corresponding PoS tags for the specified dataset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param data: dataset samples
    :type data: pd.DataFrame
    """

    if dataset_name in ['sarcasm', 'olid']:
        tweet_tokenizer = TweetTokenizer()
        tokenize_func = tweet_tokenizer.tokenize
    elif dataset_name in ['shakespeare', 'paradetox', 'politeness']:
        whitespace_tokenizer = WhitespaceTokenizer()
        tokenize_func = whitespace_tokenizer.tokenize
    else:
        tokenize_func = word_tokenize

    data['Tokens'] = data['Sentence'].apply(lambda x: tokenize_func(x.lower()))
    data['PoS Tags'] = data['Tokens'].apply(lambda x: pos_tag(x, tagset='universal', lang='eng'))
    data['Tokens with Tags'] = data['PoS Tags'].apply(lambda x: [f'{el[0]} <__> {el[1]}' for el in x])
    data.drop(['Tokens', 'PoS Tags'], axis=1, inplace=True)


def load_word_nodes(dataset_name, top=False):
    """
    Loads word nodes for the specified dataset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param top: whether to include only the 20% most discriminative words or all
    :type top: bool
    :return: word nodes
    :rtype: pd.DataFrame
    """

    dataset_name_clean = dataset_name.split('_')[0]

    word_nodes = pd.read_csv(f'{skg_root_location}/skg/{dataset_name_clean}/word.nodes', sep='\t')

    if top:
        top_k = int(len(word_nodes) * 0.2)
        word_nodes = word_nodes.nlargest(n=top_k, columns='Dist')

    return word_nodes


def load_edges(dataset_name, edge_type):
    """
    Loads edges for the specified dataset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param edge_type: edge type
    :type edge_type: str
    :return: edges
    :rtype: pd.DataFrame
    """

    dataset_name_clean = dataset_name.split('_')[0]

    edges = pd.read_csv(f'{skg_root_location}/skg/{dataset_name_clean}/{edge_type}.edges', sep='\t')

    return edges


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

    word_nodes = load_word_nodes(dataset_name=dataset_name, top=True)
    word_nodes['Node'] = word_nodes[['Word', 'Category']].apply(word_cat_merge, axis=1)
    word_nodes.drop_duplicates(subset='Node', inplace=True)
    word_nodes.set_index('Node', inplace=True)

    skg.add_nodes_from(word_nodes.index)
    nx.set_node_attributes(skg, word_nodes.to_dict(orient='index'))

    pmi_1_edges = load_edges(dataset_name=dataset_name, edge_type=f'pmi_{style_1_name}')
    pmi_1_edges['N1'] = pmi_1_edges[['W1', 'Cat1']].apply(word_cat_merge, axis=1)
    pmi_1_edges['N2'] = pmi_1_edges[['W2', 'Cat2']].apply(word_cat_merge, axis=1)
    pmi_1_edges['Attrs'] = pmi_1_edges['PMI'].apply(lambda x: {'pmi_1': x, 'edge_type': 'pmi_1_style'})
    skg.add_edges_from(pmi_1_edges[['N1', 'N2', 'Attrs']].values.tolist())

    pmi_2_edges = load_edges(dataset_name=dataset_name, edge_type=f'pmi_{style_2_name}')
    pmi_2_edges['N1'] = pmi_2_edges[['W1', 'Cat1']].apply(word_cat_merge, axis=1)
    pmi_2_edges['N2'] = pmi_2_edges[['W2', 'Cat2']].apply(word_cat_merge, axis=1)
    pmi_2_edges['Attrs'] = pmi_2_edges['PMI'].apply(lambda x: {'pmi_2': x, 'edge_type': 'pmi_2_style'})
    skg.add_edges_from(pmi_2_edges[['N1', 'N2', 'Attrs']].values.tolist())

    synonym_edges = load_edges(dataset_name=dataset_name, edge_type='synonyms')
    synonym_edges['N1'] = synonym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    synonym_edges['N2'] = synonym_edges[['Synonym', 'Synonym Category']].apply(word_cat_merge, axis=1)
    synonym_edges['Attrs'] = [{'edge_type': 'synonym'} for _ in range(len(synonym_edges))]
    skg.add_edges_from(synonym_edges[['N1', 'N2', 'Attrs']].values.tolist())

    antonym_edges = load_edges(dataset_name=dataset_name, edge_type='antonyms')
    antonym_edges['N1'] = antonym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    antonym_edges['N2'] = antonym_edges[['Antonym', 'Antonym Category']].apply(word_cat_merge, axis=1)
    antonym_edges['Attrs'] = [{'edge_type': 'antonym'} for _ in range(len(antonym_edges))]
    skg.add_edges_from(antonym_edges[['N1', 'N2', 'Attrs']].values.tolist())

    hyponym_edges = load_edges(dataset_name=dataset_name, edge_type='hyponyms')
    hyponym_edges['N1'] = hyponym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    hyponym_edges['N2'] = hyponym_edges[['Hyponym', 'Hyponym Category']].apply(word_cat_merge, axis=1)
    hyponym_edges['Attrs'] = [{'edge_type': 'hyponym'} for _ in range(len(hyponym_edges))]
    skg.add_edges_from(hyponym_edges[['N1', 'N2', 'Attrs']].values.tolist())

    hypernym_edges = load_edges(dataset_name=dataset_name, edge_type='hypernyms')
    hypernym_edges['N1'] = hypernym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    hypernym_edges['N2'] = hypernym_edges[['Hypernym', 'Hypernym Category']].apply(word_cat_merge, axis=1)
    hypernym_edges['Attrs'] = [{'edge_type': 'hypernym'} for _ in range(len(hypernym_edges))]
    skg.add_edges_from(hypernym_edges[['N1', 'N2', 'Attrs']].values.tolist())

    return skg
