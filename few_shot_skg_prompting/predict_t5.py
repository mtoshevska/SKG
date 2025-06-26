import os
import random
import pandas as pd
from argparse import ArgumentParser
from dataset import ParallelTSTDatasetFSSKGT5
from model import T5ForTextStyleTransfer

random.seed(0)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

max_length_map_maximum = {'gyafc': 256, 'paradetox': 23, 'shakespeare': 121, 'wnc': 254}

model_name_map = {'t5-small': 'T5-small',
                  't5-base': 'T5-base',
                  't5-large': 'T5-large',
                  'flan-t5-small': 'FLAN-T5-small',
                  'flan-t5-base': 'FLAN-T5-base',
                  'flan-t5-large': 'FLAN-T5-large'}

if __name__ == '__main__':
    parser = ArgumentParser('Few-shot SKG Prompting Model Arguments')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_types', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_few_shot', type=int)
    parser.add_argument('--ablation', type=int)
    parser.add_argument('--use_synonyms', type=int)
    parser.add_argument('--use_antonyms', type=int)
    parser.add_argument('--use_hypernyms', type=int)
    parser.add_argument('--use_hyponyms', type=int)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name.replace('_', '-')
    dataset_types = [dataset_type for dataset_type in args.dataset_types.split(',')]
    batch_size = args.batch_size
    n_few_shot = args.n_few_shot
    ablation = args.ablation
    if ablation:
        use_synonyms = args.use_synonyms
        use_antonyms = args.use_antonyms
        use_hypernyms = args.use_hypernyms
        use_hyponyms = args.use_hyponyms
    else:
        use_synonyms, use_antonyms, use_hypernyms, use_hyponyms = 1, 1, 1, 1

    pred_location = 'parallel' if not ablation else 'ablation'

    if not os.path.exists(f'../predictions/{pred_location}'):
        os.mkdir(f'../predictions/{pred_location}')

    if not os.path.exists(f'../predictions/{pred_location}/{dataset_name}'):
        os.mkdir(f'../predictions/{pred_location}/{dataset_name}')

    if not os.path.exists(f'../predictions/{pred_location}/{dataset_name}/{model_name}'):
        os.mkdir(f'../predictions/{pred_location}/{dataset_name}/{model_name}')

    if 'clean' in dataset_name:
        train_dataset = None
        val_dataset = None
    else:
        train_dataset = ParallelTSTDatasetFSSKGT5(dataset_name=dataset_name,
                                                  dataset_type='train',
                                                  tokenizer_name=model_name,
                                                  padding_size=max_length_map_maximum[dataset_name] * (
                                                          1 + n_few_shot) + 30,
                                                  n_few_shot=n_few_shot,
                                                  use_synonyms=use_synonyms,
                                                  use_antonyms=use_antonyms,
                                                  use_hypernyms=use_hypernyms,
                                                  use_hyponyms=use_hyponyms)
        val_dataset = ParallelTSTDatasetFSSKGT5(dataset_name=dataset_name,
                                                dataset_type='val',
                                                tokenizer_name=model_name,
                                                padding_size=max_length_map_maximum[dataset_name] * (
                                                        1 + n_few_shot) + 30,
                                                n_few_shot=n_few_shot,
                                                use_synonyms=use_synonyms,
                                                use_antonyms=use_antonyms,
                                                use_hypernyms=use_hypernyms,
                                                use_hyponyms=use_hyponyms)

    test_dataset = ParallelTSTDatasetFSSKGT5(dataset_name=dataset_name,
                                             dataset_type='test',
                                             tokenizer_name=model_name,
                                             padding_size=max_length_map_maximum[dataset_name] * (1 + n_few_shot) + 30,
                                             n_few_shot=n_few_shot,
                                             use_synonyms=use_synonyms,
                                             use_antonyms=use_antonyms,
                                             use_hypernyms=use_hypernyms,
                                             use_hyponyms=use_hyponyms)

    model = T5ForTextStyleTransfer(model_name=model_name, learning_rate=0.0001,
                                   weight_decay=0.0005, batch_size=batch_size, gpu_device='cuda',
                                   train_dataset=train_dataset, val_dataset=val_dataset,
                                   test_dataset=test_dataset, to_log=False)

    for dataset_type in dataset_types:
        print(f'Model name: {model_name}\n'
              f'Transfer direction: s1tos2\n'
              f'Dataset name: {dataset_name}\n'
              f'Dataset type: {dataset_type}\n')
        if ablation:
            print(f'Syns: {use_synonyms}\n'
                  f'Ants: {use_antonyms}\n'
                  f'Hypers: {use_hypernyms}\n'
                  f'Hypos: {use_hyponyms}\n')

        originals, gt_sentences, pred_sentences = model.predict(dataset_type,
                                                                test_dataset.tokenizer,
                                                                max_length_map_maximum[dataset_name])

        preds_df = pd.DataFrame()
        preds_df['Original'] = originals
        preds_df['Ground Truth'] = gt_sentences
        preds_df['Generated'] = pred_sentences

        primary_file_name = f'{model_name_map[model_name]}-{n_few_shot}-shot_SKG_{dataset_type}'
        ablation_str = f'Syn{use_synonyms}_Ant{use_antonyms}_Hyper{use_hypernyms}_Hypo{use_hyponyms}'

        preds_file_name = f'{primary_file_name}' if not ablation else f'{primary_file_name}_{ablation_str}'

        preds_df.to_csv(f'../predictions/{pred_location}/{dataset_name}/{model_name}/{preds_file_name}.txt',
                        sep='\t', index=False)
