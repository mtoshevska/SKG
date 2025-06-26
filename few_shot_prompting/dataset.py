import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

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


class ParallelTSTDatasetFSPT5(Dataset):
    """
    Class for dataset suitable for style transfer models based on Transformer.
    """

    def __init__(self, dataset_name, dataset_type, tokenizer_name, padding_size, n_few_shot):
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
        :param n_few_shot: number of samples for the few-shot setting
        :type n_few_shot: None or int
        """

        super(ParallelTSTDatasetFSPT5, self).__init__()

        assert dataset_name in ['gyafc', 'paradetox', 'shakespeare', 'wnc']
        assert dataset_type in ['train', 'val', 'test']

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.tokenizer_name = tokenizer_name
        self.padding_size = padding_size
        self.n_few_shot = n_few_shot

        self.__load_data__()
        self.__load_style_corpus__()
        self.__prepare_data__()

    def __load_data__(self):
        """
        Load data for the dataset.
        """

        self.data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/{self.dataset_type}_en.txt',
                                sep='\t')

    def __load_style_corpus__(self):
        """
        Load style corpus for the dataset.
        """

        if 'clean' in self.dataset_name:
            data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/test_en.txt',
                               sep='\t')
        else:
            train_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/train_en.txt',
                                     sep='\t')
            val_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/val_en.txt',
                                   sep='\t')
            test_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/test_en.txt',
                                    sep='\t')
            data = pd.concat((train_data, val_data, test_data))

        self.s1_corpus = data['Style 1'].values.tolist()
        self.s2_corpus = data['Style 2'].values.tolist()

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
        style_corpus_from = self.s1_corpus
        style_corpus_to = self.s2_corpus

        s_from = styles_map[self.dataset_name][style_from]
        s_to = styles_map[self.dataset_name][style_to]

        self.prompt_t5 = f'Paraphrase from {s_from} to {s_to}: '
        self.prefix_t5 = f'{s_from}: '
        self.suffix_t5 = f'{s_to}: '

        model = SentenceTransformer('all-distilroberta-v1')
        style_corpus_embeddings = model.encode(style_corpus_from, batch_size=128)

        in_sentence_embeddings = model.encode(self.data[style_from], convert_to_tensor=True, batch_size=128)
        results = util.semantic_search(in_sentence_embeddings, style_corpus_embeddings, top_k=self.n_few_shot + 1)

        samples = [[(style_corpus_from[r['corpus_id']],
                     style_corpus_to[r['corpus_id']]) for r in result[1:]] for result in results]

        if 'flan' in self.tokenizer_name:
            samples = [[f'{self.prompt_t5}{s[0]}\n{self.suffix_t5}{s[1]}' for s in sample] for sample in samples]
            samples = ['\n'.join(sample) for sample in samples]
            input_data = [f'{s}\n{self.prompt_t5}{i}\n{self.suffix_t5}' for i, s in
                          zip(self.data[style_from].values.tolist(), samples)]
        else:
            samples = [[f'{self.prefix_t5}{s[0]}\n{self.suffix_t5}{s[1]}' for s in sample] for sample in samples]
            samples = ['\n'.join(sample) for sample in samples]
            input_data = [f'{s}\n{self.prompt_t5}{i} </s>' for i, s in
                          zip(self.data[style_from].values.tolist(), samples)]

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


class ParallelTSTDatasetFSPLlama(Dataset):
    """
    Class for dataset suitable for style transfer models based on Llama.
    """

    def __init__(self, dataset_name, dataset_type, tokenizer_name, padding_size, n_few_shot):
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
        :param n_few_shot: number of samples for the few-shot setting
        :type n_few_shot: None or int
        """

        super(ParallelTSTDatasetFSPLlama, self).__init__()

        assert dataset_name in ['gyafc', 'paradetox', 'shakespeare', 'wnc']
        assert dataset_type in ['train', 'val', 'test']

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.tokenizer_name = tokenizer_name
        self.padding_size = padding_size
        self.n_few_shot = n_few_shot

        self.__load_data__()
        self.__load_style_corpus__()
        self.__prepare_data__()

    def __load_data__(self):
        """
        Load data for the dataset.
        """

        self.data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/{self.dataset_type}_en.txt',
                                sep='\t')

    def __load_style_corpus__(self):
        """
        Load style corpus for the dataset.
        """

        if 'clean' in self.dataset_name:
            data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/test_en.txt',
                               sep='\t')
        else:
            train_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/train_en.txt',
                                     sep='\t')
            val_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/val_en.txt',
                                   sep='\t')
            test_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/test_en.txt',
                                    sep='\t')
            data = pd.concat((train_data, val_data, test_data))

        self.s1_corpus = data['Style 1'].values.tolist()
        self.s2_corpus = data['Style 2'].values.tolist()

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
        style_corpus_from = self.s1_corpus
        style_corpus_to = self.s2_corpus

        s_from = styles_map[self.dataset_name][style_from]
        s_to = styles_map[self.dataset_name][style_to]

        model = SentenceTransformer('all-distilroberta-v1')
        style_corpus_embeddings = model.encode(style_corpus_from, batch_size=128)

        in_sentence_embeddings = model.encode(self.data[style_from], convert_to_tensor=True, batch_size=128)
        results = util.semantic_search(in_sentence_embeddings, style_corpus_embeddings, top_k=self.n_few_shot + 1)

        samples = [[(style_corpus_from[r['corpus_id']],
                     style_corpus_to[r['corpus_id']]) for r in result[1:]] for result in results]

        samples = [[f'Input: {s[0]}\n' \
                    f'Output: {s[1]}\n' for s in sample] for sample in
                   samples]
        samples = ['\n'.join(sample) for sample in samples]

        transform_func = lambda x: f'Input: {x}\n' \
                                   f'Output: '
        input_data_x = self.data[style_from].apply(transform_func).values.tolist()

        input_data = [f'{task_defs[self.dataset_name]}\n\nExamples:\n\n{s}\n\n{i}' for i, s in
                      zip(input_data_x, samples)]

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


class ParallelTSTDatasetFSPGPT(Dataset):
    """
    Class for dataset suitable for style transfer models based on GPT.
    """

    def __init__(self, dataset_name, dataset_type, tokenizer_name, padding_size, n_few_shot):
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
        :param n_few_shot: number of samples for the few-shot setting
        :type n_few_shot: None or int
        """

        super(ParallelTSTDatasetFSPGPT, self).__init__()

        assert dataset_name in ['gyafc', 'paradetox', 'shakespeare', 'wnc']
        assert dataset_type in ['train', 'val', 'test']

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.tokenizer_name = tokenizer_name
        self.padding_size = padding_size
        self.n_few_shot = n_few_shot

        self.__load_data__()
        self.__load_style_corpus__()
        self.__prepare_data__()

    def __load_data__(self):
        """
        Load data for the dataset.
        """

        self.data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/{self.dataset_type}_en.txt',
                                sep='\t')

    def __load_style_corpus__(self):
        """
        Load style corpus for the dataset.
        """

        if 'clean' in self.dataset_name:
            data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/test_en.txt',
                               sep='\t')
        else:
            train_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/train_en.txt',
                                     sep='\t')
            val_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/val_en.txt',
                                   sep='\t')
            test_data = pd.read_csv(f'{data_root_location}/parallel/{self.dataset_name}/test_en.txt',
                                    sep='\t')
            data = pd.concat((train_data, val_data, test_data))

        self.s1_corpus = data['Style 1'].values.tolist()
        self.s2_corpus = data['Style 2'].values.tolist()

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
        style_corpus_from = self.s1_corpus
        style_corpus_to = self.s2_corpus

        s_from = styles_map[self.dataset_name][style_from]
        s_to = styles_map[self.dataset_name][style_to]

        model = SentenceTransformer('all-distilroberta-v1')
        style_corpus_embeddings = model.encode(style_corpus_from, batch_size=128)

        in_sentence_embeddings = model.encode(self.data[style_from], convert_to_tensor=True, batch_size=128)
        results = util.semantic_search(in_sentence_embeddings, style_corpus_embeddings, top_k=self.n_few_shot + 1)

        samples = [[(style_corpus_from[r['corpus_id']],
                     style_corpus_to[r['corpus_id']]) for r in result[1:]] for result in results]

        samples = [[f'Here is a text, which {style_clues_for_gpt[self.dataset_name][0]}: {s[0]}\n' \
                    f'Here is a rewrite of the text, which {style_clues_for_gpt[self.dataset_name][1]}: {s[1]}\n' for s
                    in sample] for sample in samples]
        samples = ['\n'.join(sample) for sample in samples]

        transform_func = lambda x: f'Here is a text, which {style_clues_for_gpt[self.dataset_name][0]}: {x}\n' \
                                   f'Here is a rewrite of the text, which {style_clues_for_gpt[self.dataset_name][1]}: '

        input_data_x = self.data[style_from].apply(transform_func).values.tolist()

        input_data = [f'{s}\n\n{i}' for i, s in zip(input_data_x, samples)]

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
