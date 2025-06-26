import os
import numpy as np
import pandas as pd
from math import log
from tqdm import tqdm
from nltk.corpus import wordnet
from itertools import combinations
from create_nodes import load_samples, load_word_nodes, tokenize, tag_tokens


def create_co_occurrence_with_pmi_edge(dataset_name, style_name, sliding_window_size=3):
    """
    Create edges between pairs of word nodes based on PMI for the particular style.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param style_name: style
    :type style_name: str
    :param sliding_window_size: sliding window size
    :type sliding_window_size: int
    """

    if os.path.exists(f'../data/skg/{dataset_name}/pmi_{style_name}.edges'):
        return

    print(f'Creating PMI edges for style {style_name}...')

    train = load_samples(dataset_name, 'train')
    val = load_samples(dataset_name, 'val')
    test = load_samples(dataset_name, 'test')

    data = pd.concat((train, val, test))

    tokenize(dataset_name, data)
    tag_tokens(data)

    word_nodes = load_word_nodes(dataset_name, True)
    word_nodes['Token'] = word_nodes[['Word', 'Category']].apply(lambda x: f'{x[0]} <__> {x[1]}', axis=1)

    vocabulary = word_nodes['Token'].values.tolist()
    word_to_id = {vocabulary[i]: i for i in range(len(vocabulary))}

    num_windows = 0
    num_windows_i = np.zeros(len(vocabulary))
    num_windows_i_j = np.zeros((len(vocabulary), len(vocabulary)))

    tokens_all = data[data['Style'] == style_name]['Tokens with Tags'].values.tolist()

    for tokens in tqdm(tokens_all, total=len(tokens_all)):
        for window in range(max(1, len(tokens) - sliding_window_size)):
            num_windows += 1
            window_words = set(tokens[window:(window + sliding_window_size)])

            for word in window_words:
                word_id = word_to_id.get(word)
                if word_id is not None:
                    num_windows_i[word_id] += 1

            for word1, word2 in combinations(window_words, 2):
                word_id_1 = word_to_id.get(word1)
                word_id_2 = word_to_id.get(word2)
                if word_id_1 is not None and word_id_2 is not None:
                    num_windows_i_j[word_id_1][word_id_2] += 1
                    num_windows_i_j[word_id_2][word_id_1] += 1

    p_i_j_all = num_windows_i_j / num_windows
    p_i_all = num_windows_i / num_windows

    word_word_edges = []
    for word1, word2 in tqdm(combinations(vocabulary, 2), total=len([c for c in combinations(vocabulary, 2)])):
        p_i_j = p_i_j_all[word_to_id[word1]][word_to_id[word2]]
        p_i = p_i_all[word_to_id[word1]]
        p_j = p_i_all[word_to_id[word2]]
        val = log(p_i_j / (p_i * p_j)) if p_i * p_j > 0 and p_i_j > 0 else 0
        if val > 0:
            word_word_edges.append((word1, word2, val))

    word_word_edges_df = pd.DataFrame(word_word_edges, columns=['Token 1', 'Token 2', 'PMI'])

    word_word_edges_df[['W1', 'Cat1']] = word_word_edges_df['Token 1'].str.split(' <__> ', expand=True)
    word_word_edges_df.drop(['Token 1'], axis=1, inplace=True)

    word_word_edges_df[['W2', 'Cat2']] = word_word_edges_df['Token 2'].str.split(' <__> ', expand=True)
    word_word_edges_df.drop(['Token 2'], axis=1, inplace=True)

    word_word_edges_df[['W1', 'Cat1', 'W2', 'Cat2', 'PMI']].to_csv(f'../data/skg/{dataset_name}/pmi_{style_name}.edges',
                                                                   sep='\t', index=False)


def map_category_to_wordnet_pos(tag):
    """
    For a given universal PoS tag returns the corresponding WordNet PoS tag.
    :param tag: universal PoS tag
    :type tag: str
    :return: WordNet PoS tag
    :rtype: str
    """

    if tag == 'NOUN':
        return wordnet.NOUN
    elif tag == 'VERB':
        return wordnet.VERB
    elif tag == 'ADJ':
        return wordnet.ADJ
    elif tag == 'ADV':
        return wordnet.ADV
    else:
        return ''


def map_wordnet_pos_to_category(tag):
    """
    For a given corresponding WordNet PoS tag returns the universal PoS tag.
    :param tag: WordNet PoS tag
    :type tag: str
    :return: universal PoS tag
    :rtype: str
    """

    if tag == wordnet.NOUN:
        return 'NOUN'
    elif tag == wordnet.VERB:
        return 'VERB'
    elif tag == wordnet.ADJ or tag == wordnet.ADJ_SAT:
        return 'ADJ'
    elif tag == wordnet.ADV:
        return 'ADV'
    else:
        return ''


def find_synonyms(dataset_name):
    """
    Find synonyms for each word and create synonym edges.
    :param dataset_name: dataset name
    :type dataset_name: str
    """

    if os.path.exists(f'../data/skg/{dataset_name}/synonyms.edges'):
        return

    print('Creating synonym edges...')

    word_nodes = load_word_nodes(dataset_name, True)
    words = word_nodes['Word'].values.tolist()
    categories = word_nodes['Category'].values.tolist()

    synonyms = []

    for word, category in zip(words, categories):
        wordnet_pos = map_category_to_wordnet_pos(category)
        synsets = wordnet.synsets(str(word), pos=wordnet_pos)
        for synset in synsets:
            for lemma in synset.lemmas():
                if lemma.name() != word:
                    synonyms.append([word, category,
                                     lemma.name(), map_wordnet_pos_to_category(lemma._synset._pos)])

    synonyms_df = pd.DataFrame(synonyms,
                               columns=['Word', 'Category', 'Synonym', 'Synonym Category'])
    synonyms_df.drop_duplicates(inplace=True)

    synonyms_df.to_csv(f'../data/skg/{dataset_name}/synonyms.edges',
                       sep='\t', index=False)


def find_antonyms(dataset_name):
    """
    Find antonyms for each word and create antonym edges.
    :param dataset_name: dataset name
    :type dataset_name: str
    """

    if os.path.exists(f'../data/skg/{dataset_name}/antonyms.edges'):
        return

    print('Creating antonym edges...')

    word_nodes = load_word_nodes(dataset_name, True)
    words = word_nodes['Word'].values.tolist()
    categories = word_nodes['Category'].values.tolist()

    antonyms = []

    for word, category in zip(words, categories):
        wordnet_pos = map_category_to_wordnet_pos(category)
        synsets = wordnet.synsets(str(word), pos=wordnet_pos)
        for synset in synsets:
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    if antonym.name() != word:
                        antonyms.append([word, category,
                                         antonym.name(), map_wordnet_pos_to_category(antonym._synset._pos)])

    antonyms_df = pd.DataFrame(antonyms,
                               columns=['Word', 'Category', 'Antonym', 'Antonym Category'])
    antonyms_df.drop_duplicates(inplace=True)

    antonyms_df.to_csv(f'../data/skg/{dataset_name}/antonyms.edges',
                       sep='\t', index=False)


def find_hyponyms(dataset_name):
    """
    Find hyponyms for each word and create hyponym edges.
    :param dataset_name: dataset name
    :type dataset_name: str
    """

    if os.path.exists(f'../data/skg/{dataset_name}/hyponyms.edges'):
        return

    print('Creating hyponym edges...')

    word_nodes = load_word_nodes(dataset_name, True)
    words = word_nodes['Word'].values.tolist()
    categories = word_nodes['Category'].values.tolist()

    hyponyms = []

    for word, category in zip(words, categories):
        wordnet_pos = map_category_to_wordnet_pos(category)
        synsets = wordnet.synsets(str(word), pos=wordnet_pos)
        for synset in synsets:
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    if lemma.name() != word:
                        hyponyms.append([word, category,
                                         lemma.name(), map_wordnet_pos_to_category(lemma._synset._pos)])

    hyponyms_df = pd.DataFrame(hyponyms,
                               columns=['Word', 'Category', 'Hyponym', 'Hyponym Category'])
    hyponyms_df.drop_duplicates(inplace=True)

    hyponyms_df.to_csv(f'../data/skg/{dataset_name}/hyponyms.edges',
                       sep='\t', index=False)


def find_hypernyms(dataset_name):
    """
    Find hypernyms for each word and create hypernym edges.
    :param dataset_name: dataset name
    :type dataset_name: str
    """

    if os.path.exists(f'../data/skg/{dataset_name}/hypernyms.edges'):
        return

    print('Creating hypernym edges...')

    word_nodes = load_word_nodes(dataset_name, True)
    words = word_nodes['Word'].values.tolist()
    categories = word_nodes['Category'].values.tolist()

    hypernyms = []

    for word, category in zip(words, categories):
        wordnet_pos = map_category_to_wordnet_pos(category)
        synsets = wordnet.synsets(str(word), pos=wordnet_pos)
        for synset in synsets:
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    if lemma.name() != word:
                        hypernyms.append([word, category,
                                          lemma.name(), map_wordnet_pos_to_category(lemma._synset._pos)])

    hypernyms_df = pd.DataFrame(hypernyms,
                                columns=['Word', 'Category', 'Hypernym', 'Hypernym Category'])
    hypernyms_df.drop_duplicates(inplace=True)

    hypernyms_df.to_csv(f'../data/skg/{dataset_name}/hypernyms.edges',
                        sep='\t', index=False)


def load_edges(dataset_name, edge_type, words=None):
    """
    Loads edges for the specified dataset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param edge_type: edge type
    :type edge_type: str
    :return: edges
    :rtype: pd.DataFrame
    """

    edges = pd.read_csv(f'../data/skg/{dataset_name}/{edge_type}.edges', sep='\t')

    if words is not None:
        words = words[['Word', 'Category']]
        column_names = edges.columns.values
        edges = edges.merge(words, left_on=[column_names[0], column_names[1]],
                            right_on=['Word', 'Category'])
        edges = edges.merge(words, left_on=[column_names[2], column_names[3]],
                            right_on=['Word', 'Category'])
        new_columns = [f'{column_names[0]}_x', f'{column_names[1]}_x',
                       column_names[2], column_names[3]] if 'pmi' not in edge_type else column_names
        edges = edges[new_columns]
        edges.columns = column_names
    return edges
