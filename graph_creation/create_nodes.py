import os
import numpy as np
import pandas as pd
import _pickle as pickle
from nltk.tag import pos_tag
from collections import Counter
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer, WhitespaceTokenizer, word_tokenize

data_root_location = '../data'


def load_samples(dataset_name, subset_name):
    """
    Load samples for the specified dataset and subset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param subset_name: train, val, or test
    :type subset_name: str
    :return: subset samples for the specified style
    :rtype: pd.DataFrame
    """

    data = pd.read_csv(f'{data_root_location}/non-parallel/{dataset_name}/{subset_name}_en.txt',
                       sep='\t', usecols=['Sentence', 'Style'])

    return data


def tokenize(dataset_name, data):
    """
    Tokenize samples for the specified dataset.
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


def tag_tokens(data):
    """
    Determine PoS tags for the specified tokenized dataset.
    :param data: dataset samples
    :type data: pd.DataFrame
    """

    data['PoS Tags'] = data['Tokens'].apply(lambda x: pos_tag(x, tagset='universal', lang='eng'))
    data['Tokens with Tags'] = data['PoS Tags'].apply(lambda x: [f'{el[0]} <__> {el[1]}' for el in x])
    data.drop(['Tokens', 'PoS Tags'], axis=1, inplace=True)


def create_style_markers(data, style_1_name, style_2_name):
    """
    Calculate saliency scores for the specified dataset name and style names.
    :param data: dataset samples
    :type data: pd.DataFrame
    :param style_1_name: name of the first style
    :type style_1_name: str
    :param style_2_name: name of the second style
    :type style_2_name: str
    """

    data_style_0 = data.loc[data['Style'] == style_1_name]
    tokens_style_0 = [t for tokens in data_style_0['Tokens with Tags'].values.tolist() for t in tokens]
    frequencies_0 = Counter(tokens_style_0)

    data_style_1 = data.loc[data['Style'] == style_2_name]
    tokens_style_1 = [t for tokens in data_style_1['Tokens with Tags'].values.tolist() for t in tokens]
    frequencies_1 = Counter(tokens_style_1)

    saliency_0 = {token: (frequencies_0[token] + 1) / (frequencies_1[token] + 1) for token in frequencies_0}
    saliency_1 = {token: (frequencies_1[token] + 1) / (frequencies_0[token] + 1) for token in frequencies_1}

    style_0_saliency = pd.DataFrame.from_dict(saliency_0, orient='index').reset_index()
    style_0_saliency.columns = ['Word', 'S1 Saliency']

    style_1_saliency = pd.DataFrame.from_dict(saliency_1, orient='index').reset_index()
    style_1_saliency.columns = ['Word', 'S2 Saliency']

    return style_0_saliency, style_1_saliency


def create_word_nodes(dataset_name, style_1_name, style_2_name):
    """
    Create textual file with information about words, their category and saliency scores in both styles.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param style_1_name: name of the first style
    :type style_1_name: str
    :param style_2_name: name of the second style
    :type style_2_name: str
    """

    if os.path.exists(f'../data/skg/{dataset_name}/word.nodes'):
        return

    print('Creating word nodes...')

    train = load_samples(dataset_name, 'train')
    val = load_samples(dataset_name, 'val')
    test = load_samples(dataset_name, 'test')

    data = pd.concat((train, val, test))

    tokenize(dataset_name, data)
    tag_tokens(data)

    style_0_saliency, style_1_saliency = create_style_markers(data, style_1_name, style_2_name)

    all_tokens = set([t for tokens in data['Tokens with Tags'].values for t in tokens])
    tokens_df = pd.DataFrame(all_tokens, columns=['Token'])

    tokens_df = tokens_df.merge(style_0_saliency, how='left',
                                left_on='Token', right_on='Word').merge(style_1_saliency, how='left', left_on='Token',
                                                                        right_on='Word')
    tokens_df.drop(['Word_x', 'Word_y'], axis=1, inplace=True)
    tokens_df.fillna(0, inplace=True)

    tokens_df['Dist'] = abs(tokens_df['S1 Saliency'] - tokens_df['S2 Saliency'])

    tokens_df[['Word', 'Category']] = tokens_df['Token'].str.split(' <__> ', expand=True)
    tokens_df.drop(['Token'], axis=1, inplace=True)

    tokens_df[['Word', 'Category', 'S1 Saliency', 'S2 Saliency', 'Dist']].to_csv(
        f'../data/skg/{dataset_name}/word.nodes',
        sep='\t', index=False)


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

    word_nodes = pd.read_csv(f'../data/skg/{dataset_name}/word.nodes', sep='\t')

    if top:
        top_k = int(len(word_nodes) * 0.2)
        word_nodes = word_nodes.nlargest(n=top_k, columns='Dist')

    return word_nodes


def create_word_embeddings(dataset_name):
    """
    Create pickle file with the Word2Vec embedding representation of the word nodes.
    :param dataset_name: dataset name
    :type dataset_name: str
    """

    if os.path.exists(f'../data/skg/{dataset_name}/word_embeddings.pkl'):
        return

    print('Creating word embeddings...')

    train = load_samples(dataset_name, 'train')
    val = load_samples(dataset_name, 'val')
    test = load_samples(dataset_name, 'test')

    data = pd.concat((train, val, test))

    tokenize(dataset_name, data)
    tag_tokens(data)

    all_tokens = set([t for tokens in data['Tokens with Tags'].values for t in tokens])

    model = Word2Vec(data['Tokens with Tags'], vector_size=128, min_count=5, window=5, sg=1)
    words = model.wv.index_to_key
    vectors = model.wv.vectors

    word_embeddings = {word: vector for word, vector in zip(words, vectors)}
    for word in set(all_tokens).difference(set(words)):
        word_embeddings[word] = np.random.normal(size=128)

    with open(f'../data/skg/{dataset_name}/word_embeddings.pkl', 'wb') as doc:
        pickle.dump(word_embeddings, doc)
