import os
import random
import pandas as pd
from huggingface_hub import login
from argparse import ArgumentParser
from dataset import ParallelTSTDatasetZSPLlama
from model import LlamaForTextStyleTransfer

random.seed(0)

login(token='<YOUR_TOKEN_HERE>')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

max_length_map_maximum = {'gyafc': 256, 'paradetox': 23, 'shakespeare': 121, 'wnc': 254}

model_name_map = {'llama-7b': 'Llama-2-7b-hf',
                  'llama-13b': 'Llama-2-13b-hf',
                  'llama-7b-chat': 'Llama-2-7b-chat-hf',
                  'llama-13b-chat': 'Llama-2-13b-chat-hf'}

if __name__ == '__main__':
    parser = ArgumentParser('Zero-shot Prompt Model Arguments')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_types', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_few_shot', type=int)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name.replace('_', '-')
    dataset_types = [dataset_type for dataset_type in args.dataset_types.split(',')]
    batch_size = args.batch_size
    n_few_shot = args.n_few_shot

    if not os.path.exists(f'../predictions/parallel/{dataset_name}'):
        os.mkdir(f'../predictions/parallel/{dataset_name}')

    if not os.path.exists(f'../predictions/parallel/{dataset_name}/{model_name}'):
        os.mkdir(f'../predictions/parallel/{dataset_name}/{model_name}')

    train_dataset = ParallelTSTDatasetZSPLlama(dataset_name=dataset_name,
                                               dataset_type='train',
                                               tokenizer_name=model_name_map[model_name],
                                               padding_size=max_length_map_maximum[dataset_name] + 100)
    val_dataset = ParallelTSTDatasetZSPLlama(dataset_name=dataset_name,
                                             dataset_type='val',
                                             tokenizer_name=model_name_map[model_name],
                                             padding_size=max_length_map_maximum[dataset_name] + 100)
    test_dataset = ParallelTSTDatasetZSPLlama(dataset_name=dataset_name,
                                              dataset_type='test',
                                              tokenizer_name=model_name_map[model_name],
                                              padding_size=max_length_map_maximum[dataset_name] + 100)

    model = LlamaForTextStyleTransfer(model_name=model_name_map[model_name], batch_size=batch_size,
                                      train_dataset=train_dataset, val_dataset=val_dataset,
                                      test_dataset=test_dataset, to_log=False)

    for dataset_type in dataset_types:
        print(f'Model name: {model_name}\n'
              f'Transfer direction: s1tos2\n'
              f'Dataset name: {dataset_name}\n'
              f'Dataset type: {dataset_type}\n')

        originals, gt_sentences, pred_sentences = model.predict(dataset_type,
                                                                test_dataset.tokenizer,
                                                                max_length_map_maximum[dataset_name])

        preds_df = pd.DataFrame()
        preds_df['Original'] = originals
        preds_df['Ground Truth'] = gt_sentences
        preds_df['Generated'] = pred_sentences

        preds_df.to_csv(
            f'../predictions/parallel/{dataset_name}/{model_name}/{model_name_map[model_name]}-0-shot_{dataset_type}.txt',
            sep='\t', index=False)
