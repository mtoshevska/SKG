import os
import random
import pandas as pd
from argparse import ArgumentParser
from dataset import ParallelTSTDatasetZSPGPT
from model import GPTForTextStyleTransfer

random.seed(0)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

max_length_map_maximum = {'gyafc': 256, 'paradetox': 23, 'shakespeare': 121, 'wnc': 254}

model_name_map = {'gpt-j-6b': 'GPT-J-6B',
                  'gpt-neo-1.3b': 'GPT-Neo-1.3B'}

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

    train_dataset = ParallelTSTDatasetZSPGPT(dataset_name=dataset_name,
                                             dataset_type='train',
                                             tokenizer_name=model_name,
                                             padding_size=max_length_map_maximum[dataset_name] + 30)
    val_dataset = ParallelTSTDatasetZSPGPT(dataset_name=dataset_name,
                                           dataset_type='val',
                                           tokenizer_name=model_name,
                                           padding_size=max_length_map_maximum[dataset_name] + 30)
    test_dataset = ParallelTSTDatasetZSPGPT(dataset_name=dataset_name,
                                            dataset_type='test',
                                            tokenizer_name=model_name,
                                            padding_size=max_length_map_maximum[dataset_name] + 30)

    model = GPTForTextStyleTransfer(model_name=model_name, learning_rate=0.0001,
                                    weight_decay=0.0005, batch_size=batch_size, gpu_device='cuda',
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
