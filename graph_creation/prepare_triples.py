import pandas as pd
from create_edges import load_edges
from argparse import ArgumentParser


def prepare_triples(dataset_name, style_1_name, style_2_name):
    """
    Creates and writes KG triples suitable for training KGE models for the given dataset.
    :param dataset_name: dataset name
    :type dataset_name: str
    :param style_1_name: style 1
    :type style_1_name: str
    :param style_2_name: style 2
    :type style_2_name: str
    """

    triples = []
    word_cat_merge = lambda x: f'{x[0]} <__> {x[1]}'

    pmi_1_edges = load_edges(dataset_name=dataset_name, edge_type=f'pmi_{style_1_name}')
    pmi_1_edges['N1'] = pmi_1_edges[['W1', 'Cat1']].apply(word_cat_merge, axis=1)
    pmi_1_edges['N2'] = pmi_1_edges[['W2', 'Cat2']].apply(word_cat_merge, axis=1)
    pmi_1_edges['E'] = ['pmi_1_style' for _ in range(len(pmi_1_edges))]
    triples.extend(pmi_1_edges[['N1', 'E', 'N2']].values.tolist())

    pmi_2_edges = load_edges(dataset_name=dataset_name, edge_type=f'pmi_{style_2_name}')
    pmi_2_edges['N1'] = pmi_2_edges[['W1', 'Cat1']].apply(word_cat_merge, axis=1)
    pmi_2_edges['N2'] = pmi_2_edges[['W2', 'Cat2']].apply(word_cat_merge, axis=1)
    pmi_2_edges['E'] = ['pmi_2_style' for _ in range(len(pmi_2_edges))]
    triples.extend(pmi_2_edges[['N1', 'E', 'N2']].values.tolist())

    synonym_edges = load_edges(dataset_name=dataset_name, edge_type='synonyms')
    synonym_edges['N1'] = synonym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    synonym_edges['N2'] = synonym_edges[['Synonym', 'Synonym Category']].apply(word_cat_merge, axis=1)
    synonym_edges['E'] = ['synonym' for _ in range(len(synonym_edges))]
    triples.extend(synonym_edges[['N1', 'E', 'N2']].values.tolist())

    antonym_edges = load_edges(dataset_name=dataset_name, edge_type='antonyms')
    antonym_edges['N1'] = antonym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    antonym_edges['N2'] = antonym_edges[['Antonym', 'Antonym Category']].apply(word_cat_merge, axis=1)
    antonym_edges['E'] = ['antonym' for _ in range(len(antonym_edges))]
    triples.extend(antonym_edges[['N1', 'E', 'N2']].values.tolist())

    hyponym_edges = load_edges(dataset_name=dataset_name, edge_type='hyponyms')
    hyponym_edges['N1'] = hyponym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    hyponym_edges['N2'] = hyponym_edges[['Hyponym', 'Hyponym Category']].apply(word_cat_merge, axis=1)
    hyponym_edges['E'] = ['hyponym' for _ in range(len(hyponym_edges))]
    triples.extend(hyponym_edges[['N1', 'E', 'N2']].values.tolist())

    hypernym_edges = load_edges(dataset_name=dataset_name, edge_type='hypernyms')
    hypernym_edges['N1'] = hypernym_edges[['Word', 'Category']].apply(word_cat_merge, axis=1)
    hypernym_edges['N2'] = hypernym_edges[['Hypernym', 'Hypernym Category']].apply(word_cat_merge, axis=1)
    hypernym_edges['E'] = ['hypernym' for _ in range(len(hypernym_edges))]
    triples.extend(hypernym_edges[['N1', 'E', 'N2']].values.tolist())

    triples_df = pd.DataFrame(triples, columns=['Entity 1', 'Relation', 'Entity 2'])
    triples_df.to_csv(f'../data/skg/{dataset_name}/triples.tsv',
                      sep='\t', index=False)


if __name__ == '__main__':
    parser = ArgumentParser('Style KG Arguments')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--style_1_name', type=str)
    parser.add_argument('--style_2_name', type=str)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    style_1_name = args.style_1_name
    style_2_name = args.style_2_name

    prepare_triples(dataset_name=dataset_name, style_1_name=style_1_name, style_2_name=style_2_name)
