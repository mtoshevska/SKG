import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from skg_utils import tokenize_and_tag, load_word_nodes, word_cat_merge, load_edges

data_root_location = '../data'

styles_map = {
    'gyafc': {'Style 1': 'informal', 'Style 2': 'formal'},
    'paradetox': {'Style 1': 'toxic', 'Style 2': 'neutral'},
    'shakespeare': {'Style 1': 'modern', 'Style 2': 'shakespearean'},
    'wnc': {'Style 1': 'biased', 'Style 2': 'neutral'}
}

task_defs = {
    'gyafc': 'Formality transfer changes an informal sentence to a formal sentence while keeping its general meaning unchanged.',
    'wnc': 'Neutralizing subjective bias changes a sentence that contains subjective bias to a clean sentence while keeping its general meaning unchanged.',
    'shakespeare': 'Shakespearizing modern English changes a sentence written in modern English to a sentence written in Shakespearean English while keeping its general meaning unchanged.',
    'paradetox': 'Text detoxification changes a toxic sentence to a clean sentence while keeping its general meaning unchanged.'
}

task_clues = {
    'gyafc': 'transfer the formality of the following English sentence.',
    'wnc': 'neutralize the following English sentence.',
    'shakespeare': 'Shakespearize the following English sentence.',
    'paradetox': 'detoxify the following English sentence.'
}

style_clues_for_gpt = {
    'gyafc': ['is informal', 'is formal'],
    'wnc': ['contains subjective bias', 'is neutral'],
    'shakespeare': ['is in modern English', 'is in Shakespearean English'],
    'paradetox': ['contains toxic language', 'is neutral']
}


class ParallelTSTDatasetZSSKGT5(Dataset):
    """
    Class for dataset suitable for style transfer models based on Transformer.
    """

    def __init__(self, dataset_name, dataset_type, tokenizer_name, padding_size):
        """
        Initialize dataset class.
        :param dataset_name: dataset name
        :type dataset_name: str
        :param dataset_type: train, val, or test
        :type dataset_type: str
        :param tokenizer_name: name of the Transformer-based tokenizer
        :type tokenizer_name: str
        :param padding_size: padding size
        :type padding_size: int
        """

        super(ParallelTSTDatasetZSSKGT5, self).__init__()

        assert dataset_name in ['gyafc', 'paradetox', 'shakespeare', 'wnc']
        assert dataset_type in ['train', 'val', 'test']

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.tokenizer_name = tokenizer_name
        self.padding_size = padding_size

        self.__load_data__()
        self.__prepare_data__()

    def __load_data__(self):
        """
        Load data for the dataset.
        """

        self.data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/{self.dataset_type}_en.txt',
                                sep='\t')

    def __prepare_data__(self):
        """
        Prepare samples for training/evaluation.
        """

        if 'flan' in self.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(f'google/{self.tokenizer_name}',
                                                           model_max_length=self.padding_size)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name,
                                                           model_max_length=self.padding_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs, inputs_wo_prefix, outputs = self.__prepare_data_parallel__()

        x_data_input_ids = inputs['input_ids']
        x_data_attention_masks = inputs['attention_mask']
        x_data_wo_prefix_input_ids = inputs_wo_prefix['input_ids']
        y_data_input_ids = outputs['input_ids']

        self.x_data_input_ids = torch.from_numpy(np.array(x_data_input_ids))
        self.x_data_attention_masks = torch.from_numpy(np.array(x_data_attention_masks))
        self.x_data_wo_prefix_input_ids = torch.from_numpy(np.array(x_data_wo_prefix_input_ids))
        self.y_data_input_ids = torch.from_numpy(np.array(y_data_input_ids))
        self.number_of_samples = len(x_data_input_ids)

    def __prepare_data_parallel__(self):
        """
        Prepare samples for training/evaluation for a parallel dataset.
        :return: inputs and outputs for the model
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor
        """

        style_from = 'Style 1'
        style_to = 'Style 2'
        saliency_from = 'S1 Saliency'
        saliency_to = 'S2 Saliency'

        s_from = styles_map[self.dataset_name][style_from]
        s_to = styles_map[self.dataset_name][style_to]

        self.prompt_t5 = f'Paraphrase from {s_from} to {s_to}: '
        self.prefix_t5 = f'{s_from}: '
        self.suffix_t5 = f'{s_to}: '

        token_saliencies = load_word_nodes(dataset_name=self.dataset_name, top=True)
        token_saliencies['Token'] = token_saliencies[['Word', 'Category']].apply(word_cat_merge, axis=1)
        token_saliencies.set_index('Token', inplace=True)
        token_saliencies_dict = token_saliencies[saliency_from].to_dict()

        input_data_df = self.data[style_from].to_frame()
        input_data_df.columns = ['Sentence']
        tokenize_and_tag(self.dataset_name, input_data_df)

        input_data_df['Tokens with Saliencies'] = input_data_df['Tokens with Tags'].apply(
            lambda x: [(t, token_saliencies_dict[t]) for t in x if t in token_saliencies_dict])
        input_data_df['Top-3'] = input_data_df['Tokens with Saliencies'].apply(lambda x: sorted([t[1] for t in x])[-3:])
        input_data_df['Style Markers'] = input_data_df[['Tokens with Saliencies', 'Top-3']].apply(
            lambda x: [t[0] for t in x[0] if t[1] in x[1]],
            axis=1)
        input_data_df.drop(['Tokens with Tags', 'Tokens with Saliencies', 'Top-3'], axis=1, inplace=True)

        synonyms = load_edges(dataset_name=self.dataset_name, edge_type='synonyms')
        synonyms['N1'] = synonyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        synonyms['N2'] = synonyms[['Synonym', 'Synonym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(synonyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Synonyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        antonyms = load_edges(dataset_name=self.dataset_name, edge_type='antonyms')
        antonyms['N1'] = antonyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        antonyms['N2'] = antonyms[['Antonym', 'Antonym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(antonyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Antonyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        hyponyms = load_edges(dataset_name=self.dataset_name, edge_type='hyponyms')
        hyponyms['N1'] = hyponyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        hyponyms['N2'] = hyponyms[['Hyponym', 'Hyponym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(hyponyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Hyponyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        hypernyms = load_edges(dataset_name=self.dataset_name, edge_type='hypernyms')
        hypernyms['N1'] = hypernyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        hypernyms['N2'] = hypernyms[['Hypernym', 'Hypernym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(hypernyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Hypernyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        if 'flan' in self.tokenizer_name:
            transform_func = lambda x: f'{self.prompt_t5}{x}\n{self.suffix_t5}'
        else:
            transform_func = lambda x: f'{self.prompt_t5}{x}</s>'

        input_style_markers = input_data_df['Style Markers'].apply(
            lambda x: ', '.join(set([t.split(' <__> ')[0] for t in x]))).values.tolist()
        input_synonyms = input_data_df['Style Markers Synonyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_antonyms = input_data_df['Style Markers Antonyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_hyponyms = input_data_df['Style Markers Hyponyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_hypernyms = input_data_df['Style Markers Hypernyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_data_text = self.data[style_from].apply(transform_func).values.tolist()

        input_data = []
        for in_sm, in_syn, in_ant, in_hypo, in_hyper, in_txt in zip(input_style_markers, input_synonyms, input_antonyms,
                                                                    input_hyponyms, input_hypernyms, input_data_text):
            in_data = ''
            in_data += f'Style Markers: {in_sm}\n' if len(in_sm) > 0 else ''
            in_data += f'Synonyms: {in_syn}\n' if len(in_syn) > 0 else ''
            in_data += f'Antonyms: {in_ant}\n' if len(in_ant) > 0 else ''
            in_data += f'Hyponyms: {in_hypo}\n' if len(in_hypo) > 0 else ''
            in_data += f'Hypernyms: {in_hyper}\n' if len(in_hyper) > 0 else ''
            in_data += in_txt

            input_data.append(in_data)

        input_data_wo_prefix = self.data[style_from].values.tolist()
        output_data = self.data[style_to].values.tolist()

        inputs = self.tokenizer(input_data, truncation=True,
                                padding=True, return_tensors='pt')
        inputs_wo_prefix = self.tokenizer(input_data_wo_prefix, truncation=True,
                                          padding=True, return_tensors='pt')
        outputs = self.tokenizer(output_data, truncation=True,
                                 padding=True, return_tensors='pt')

        return inputs, inputs_wo_prefix, outputs

    def __getitem__(self, index):
        return self.x_data_input_ids[index], \
            self.x_data_attention_masks[index], \
            self.x_data_wo_prefix_input_ids[index], \
            self.y_data_input_ids[index]

    def __len__(self):
        return self.number_of_samples

    def __str__(self):
        return f'Dataset name: {self.dataset_name}\n' \
               f'Dataset type: {self.dataset_type}\n' \
               f'Number of samples: {self.number_of_samples}\n'


class ParallelTSTDatasetZSSKGLlama(Dataset):
    """
    Class for dataset suitable for style transfer models based on Llama.
    """

    def __init__(self, dataset_name, dataset_type, tokenizer_name, padding_size):
        """
        Initialize dataset class.
        :param dataset_name: dataset name
        :type dataset_name: str
        :param dataset_type: train, val, or test
        :type dataset_type: str
        :param tokenizer_name: name of the tokenizer
        :type tokenizer_name: str
        :param padding_size: padding size
        :type padding_size: int
        """

        super(ParallelTSTDatasetZSSKGLlama, self).__init__()

        assert dataset_name in ['gyafc', 'paradetox', 'shakespeare', 'wnc']
        assert dataset_type in ['train', 'val', 'test']

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.tokenizer_name = tokenizer_name
        self.padding_size = padding_size

        self.__load_data__()
        self.__prepare_data__()

    def __load_data__(self):
        """
        Load data for the dataset.
        """

        self.data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/{self.dataset_type}_en.txt',
                                sep='\t')

    def __prepare_data__(self):
        """
        Prepare samples for training/evaluation.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(f'meta-llama/{self.tokenizer_name}',
                                                       use_auth_token=True,
                                                       model_max_length=self.padding_size)
        self.tokenizer.pad_token = self.tokenizer.bos_token
        self.tokenizer.padding_side = 'left'

        inputs, inputs_wo_prefix, outputs = self.__prepare_data_parallel__()

        x_data_input_ids = inputs['input_ids']
        x_data_attention_masks = inputs['attention_mask']
        x_data_wo_prefix_input_ids = inputs_wo_prefix['input_ids']
        y_data_input_ids = outputs['input_ids']

        self.x_data_input_ids = torch.from_numpy(np.array(x_data_input_ids))
        self.x_data_attention_masks = torch.from_numpy(np.array(x_data_attention_masks))
        self.x_data_wo_prefix_input_ids = torch.from_numpy(np.array(x_data_wo_prefix_input_ids))
        self.y_data_input_ids = torch.from_numpy(np.array(y_data_input_ids))
        self.number_of_samples = len(x_data_input_ids)

    def __prepare_data_parallel__(self):
        """
        Prepare samples for training/evaluation for a parallel dataset.
        :return: inputs and outputs for the model
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor
        """

        style_from = 'Style 1'
        style_to = 'Style 2'
        saliency_from = 'S1 Saliency'
        saliency_to = 'S2 Saliency'

        token_saliencies = load_word_nodes(dataset_name=self.dataset_name, top=True)
        token_saliencies['Token'] = token_saliencies[['Word', 'Category']].apply(word_cat_merge, axis=1)
        token_saliencies.set_index('Token', inplace=True)
        token_saliencies_dict = token_saliencies[saliency_from].to_dict()

        input_data_df = self.data[style_from].to_frame()
        input_data_df.columns = ['Sentence']
        tokenize_and_tag(self.dataset_name, input_data_df)

        input_data_df['Tokens with Saliencies'] = input_data_df['Tokens with Tags'].apply(
            lambda x: [(t, token_saliencies_dict[t]) for t in x if t in token_saliencies_dict])
        input_data_df['Top-3'] = input_data_df['Tokens with Saliencies'].apply(lambda x: sorted([t[1] for t in x])[-3:])
        input_data_df['Style Markers'] = input_data_df[['Tokens with Saliencies', 'Top-3']].apply(
            lambda x: [t[0] for t in x[0] if t[1] in x[1]],
            axis=1)
        input_data_df.drop(['Tokens with Tags', 'Tokens with Saliencies', 'Top-3'], axis=1, inplace=True)

        synonyms = load_edges(dataset_name=self.dataset_name, edge_type='synonyms')
        synonyms['N1'] = synonyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        synonyms['N2'] = synonyms[['Synonym', 'Synonym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(synonyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Synonyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        antonyms = load_edges(dataset_name=self.dataset_name, edge_type='antonyms')
        antonyms['N1'] = antonyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        antonyms['N2'] = antonyms[['Antonym', 'Antonym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(antonyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Antonyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        hyponyms = load_edges(dataset_name=self.dataset_name, edge_type='hyponyms')
        hyponyms['N1'] = hyponyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        hyponyms['N2'] = hyponyms[['Hyponym', 'Hyponym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(hyponyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Hyponyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        hypernyms = load_edges(dataset_name=self.dataset_name, edge_type='hypernyms')
        hypernyms['N1'] = hypernyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        hypernyms['N2'] = hypernyms[['Hypernym', 'Hypernym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(hypernyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Hypernyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        input_style_markers = input_data_df['Style Markers'].apply(
            lambda x: ', '.join(set([t.split(' <__> ')[0] for t in x]))).values.tolist()
        input_synonyms = input_data_df['Style Markers Synonyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_antonyms = input_data_df['Style Markers Antonyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_hyponyms = input_data_df['Style Markers Hyponyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_hypernyms = input_data_df['Style Markers Hypernyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()

        transform_func = lambda x: f'Input: {x}\n'
        input_data_text = self.data[style_from].apply(transform_func).values.tolist()

        input_data = []
        for in_sm, in_syn, in_ant, in_hypo, in_hyper, in_txt in zip(input_style_markers, input_synonyms, input_antonyms,
                                                                    input_hyponyms, input_hypernyms, input_data_text):
            in_data = f'{task_defs[self.dataset_name]}\n' \
                      f'Now {task_clues[self.dataset_name]}\n'
            in_data += in_txt
            in_data += f'Style Markers: {in_sm}\n' if len(in_sm) > 0 else ''
            in_data += f'Synonyms: {in_syn}\n' if len(in_syn) > 0 else ''
            in_data += f'Antonyms: {in_ant}\n' if len(in_ant) > 0 else ''
            in_data += f'Hyponyms: {in_hypo}\n' if len(in_hypo) > 0 else ''
            in_data += f'Hypernyms: {in_hyper}\n' if len(in_hyper) > 0 else ''
            in_data += 'Output: '

            input_data.append(in_data)

        input_data_wo_prefix = self.data[style_from].values.tolist()
        output_data = self.data[style_to].values.tolist()

        inputs = self.tokenizer(input_data, truncation=True,
                                padding=True, return_tensors='pt')
        inputs_wo_prefix = self.tokenizer(input_data_wo_prefix, truncation=True,
                                          padding=True, return_tensors='pt')
        outputs = self.tokenizer(output_data, truncation=True,
                                 padding=True, return_tensors='pt')

        return inputs, inputs_wo_prefix, outputs

    def __getitem__(self, index):
        return self.x_data_input_ids[index], \
            self.x_data_attention_masks[index], \
            self.x_data_wo_prefix_input_ids[index], \
            self.y_data_input_ids[index]

    def __len__(self):
        return self.number_of_samples

    def __str__(self):
        return f'Dataset name: {self.dataset_name}\n' \
               f'Dataset type: {self.dataset_type}\n' \
               f'Number of samples: {self.number_of_samples}\n'


class ParallelTSTDatasetZSSKGGPT(Dataset):
    """
    Class for dataset suitable for style transfer models based on GPT.
    """

    def __init__(self, dataset_name, dataset_type, tokenizer_name, padding_size):
        """
        Initialize dataset class.
        :param dataset_name: dataset name
        :type dataset_name: str
        :param dataset_type: train, val, or test
        :type dataset_type: str
        :param tokenizer_name: name of the tokenizer
        :type tokenizer_name: str
        :param padding_size: padding size
        :type padding_size: int
        """

        super(ParallelTSTDatasetZSSKGGPT, self).__init__()

        assert dataset_name in ['gyafc', 'paradetox', 'shakespeare', 'wnc']
        assert dataset_type in ['train', 'val', 'test']

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.tokenizer_name = tokenizer_name
        self.padding_size = padding_size

        self.__load_data__()
        self.__prepare_data__()

    def __load_data__(self):
        """
        Load data for the dataset.
        """

        self.data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/{self.dataset_type}_en.txt',
                                sep='\t')

    def __prepare_data__(self):
        """
        Prepare samples for training/evaluation.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(f'EleutherAI/{self.tokenizer_name}',
                                                       model_max_length=self.padding_size)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs, inputs_wo_prefix, outputs = self.__prepare_data_parallel__()

        x_data_input_ids = inputs['input_ids']
        x_data_attention_masks = inputs['attention_mask']
        x_data_wo_prefix_input_ids = inputs_wo_prefix['input_ids']
        y_data_input_ids = outputs['input_ids']

        self.x_data_input_ids = torch.from_numpy(np.array(x_data_input_ids))
        self.x_data_attention_masks = torch.from_numpy(np.array(x_data_attention_masks))
        self.x_data_wo_prefix_input_ids = torch.from_numpy(np.array(x_data_wo_prefix_input_ids))
        self.y_data_input_ids = torch.from_numpy(np.array(y_data_input_ids))
        self.number_of_samples = len(x_data_input_ids)

    def __prepare_data_parallel__(self):
        """
        Prepare samples for training/evaluation for a parallel dataset.
        :return: inputs and outputs for the model
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor
        """

        style_from = 'Style 1'
        style_to = 'Style 2'
        saliency_from = 'S1 Saliency'
        saliency_to = 'S2 Saliency'

        token_saliencies = load_word_nodes(dataset_name=self.dataset_name, top=True)
        token_saliencies['Token'] = token_saliencies[['Word', 'Category']].apply(word_cat_merge, axis=1)
        token_saliencies.set_index('Token', inplace=True)
        token_saliencies_dict = token_saliencies[saliency_from].to_dict()

        input_data_df = self.data[style_from].to_frame()
        input_data_df.columns = ['Sentence']
        tokenize_and_tag(self.dataset_name, input_data_df)

        input_data_df['Tokens with Saliencies'] = input_data_df['Tokens with Tags'].apply(
            lambda x: [(t, token_saliencies_dict[t]) for t in x if t in token_saliencies_dict])
        input_data_df['Top-3'] = input_data_df['Tokens with Saliencies'].apply(lambda x: sorted([t[1] for t in x])[-3:])
        input_data_df['Style Markers'] = input_data_df[['Tokens with Saliencies', 'Top-3']].apply(
            lambda x: [t[0] for t in x[0] if t[1] in x[1]],
            axis=1)
        input_data_df.drop(['Tokens with Tags', 'Tokens with Saliencies', 'Top-3'], axis=1, inplace=True)

        synonyms = load_edges(dataset_name=self.dataset_name, edge_type='synonyms')
        synonyms['N1'] = synonyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        synonyms['N2'] = synonyms[['Synonym', 'Synonym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(synonyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Synonyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        antonyms = load_edges(dataset_name=self.dataset_name, edge_type='antonyms')
        antonyms['N1'] = antonyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        antonyms['N2'] = antonyms[['Antonym', 'Antonym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(antonyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Antonyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        hyponyms = load_edges(dataset_name=self.dataset_name, edge_type='hyponyms')
        hyponyms['N1'] = hyponyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        hyponyms['N2'] = hyponyms[['Hyponym', 'Hyponym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(hyponyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Hyponyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        hypernyms = load_edges(dataset_name=self.dataset_name, edge_type='hypernyms')
        hypernyms['N1'] = hypernyms[['Word', 'Category']].apply(word_cat_merge, axis=1)
        hypernyms['N2'] = hypernyms[['Hypernym', 'Hypernym Category']].apply(word_cat_merge, axis=1)
        skg = nx.MultiGraph()
        skg.add_edges_from(hypernyms[['N1', 'N2']].values.tolist())
        input_data_df['Style Markers Hypernyms'] = input_data_df['Style Markers'].apply(
            lambda x: set([t1.split(' <__> ')[0] for t in x if t in skg.nodes for t1 in skg.neighbors(t)]))

        input_style_markers = input_data_df['Style Markers'].apply(
            lambda x: ', '.join(set([t.split(' <__> ')[0] for t in x]))).values.tolist()
        input_synonyms = input_data_df['Style Markers Synonyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_antonyms = input_data_df['Style Markers Antonyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_hyponyms = input_data_df['Style Markers Hyponyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()
        input_hypernyms = input_data_df['Style Markers Hypernyms'].apply(
            lambda x: ', '.join(list(set([t.split(' <__> ')[0] for t in x]))[:3])).values.tolist()

        transform_func = lambda x: f'Here is a text, which {style_clues_for_gpt[self.dataset_name][0]}: {x}\n'
        input_data_text = self.data[style_from].apply(transform_func).values.tolist()

        input_data = []
        for in_sm, in_syn, in_ant, in_hypo, in_hyper, in_txt in zip(input_style_markers, input_synonyms, input_antonyms,
                                                                    input_hyponyms, input_hypernyms, input_data_text):
            in_data = f'{in_txt}'
            in_data += f'Style Markers: {in_sm}\n' if len(in_sm) > 0 else ''
            in_data += f'Synonyms: {in_syn}\n' if len(in_syn) > 0 else ''
            in_data += f'Antonyms: {in_ant}\n' if len(in_ant) > 0 else ''
            in_data += f'Hyponyms: {in_hypo}\n' if len(in_hypo) > 0 else ''
            in_data += f'Hypernyms: {in_hyper}\n' if len(in_hyper) > 0 else ''
            in_data += f'Here is a rewrite of the text, which {style_clues_for_gpt[self.dataset_name][1]}: '

            input_data.append(in_data)

        input_data_wo_prefix = self.data[style_from].values.tolist()
        output_data = self.data[style_to].values.tolist()

        inputs = self.tokenizer(input_data, truncation=True,
                                padding=True, return_tensors='pt')
        inputs_wo_prefix = self.tokenizer(input_data_wo_prefix, truncation=True,
                                          padding=True, return_tensors='pt')
        outputs = self.tokenizer(output_data, truncation=True,
                                 padding=True, return_tensors='pt')

        return inputs, inputs_wo_prefix, outputs

    def __getitem__(self, index):
        return self.x_data_input_ids[index], \
            self.x_data_attention_masks[index], \
            self.x_data_wo_prefix_input_ids[index], \
            self.y_data_input_ids[index]

    def __len__(self):
        return self.number_of_samples

    def __str__(self):
        return f'Dataset name: {self.dataset_name}\n' \
               f'Dataset type: {self.dataset_type}\n' \
               f'Number of samples: {self.number_of_samples}\n'
