import os
import random
import numpy as np
import pandas as pd
from model_names import model_name_map
from evaluation import calculate_bleu, calculate_bertscore, calculate_accuracy, calculate_perplexity
from argparse import ArgumentParser

random.seed(0)

if __name__ == '__main__':
    parser = ArgumentParser('Evaluation Arguments')
    parser.add_argument('--dataset_names', type=str)
    parser.add_argument('--dataset_types', type=str)
    parser.add_argument('--model_names', type=str)
    parser.add_argument('--transfer_directions', type=str)
    parser.add_argument('--st_type', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--ablation', type=int)
    parser.add_argument('--use_synonyms', type=int)
    parser.add_argument('--use_antonyms', type=int)
    parser.add_argument('--use_hypernyms', type=int)
    parser.add_argument('--use_hyponyms', type=int)

    args = parser.parse_args()
    dataset_names = [dataset_name for dataset_name in args.dataset_names.split(',')]
    dataset_types = [dataset_type for dataset_type in args.dataset_types.split(',')]
    model_names = [model_name for model_name in args.model_names.split(',')]
    transfer_directions = [transfer_direction for transfer_direction in args.transfer_directions.split(',')]
    st_type = args.st_type.replace('_', '-')
    batch_size = args.batch_size
    ablation = args.ablation
    if ablation:
        use_synonyms = args.use_synonyms
        use_antonyms = args.use_antonyms
        use_hypernyms = args.use_hypernyms
        use_hyponyms = args.use_hyponyms
    else:
        use_synonyms, use_antonyms, use_hypernyms, use_hyponyms = None, None, None, None

    for dataset_name in dataset_names:
        if not os.path.exists(f'../results/{st_type}/{dataset_name}'):
            os.mkdir(f'../results/{st_type}/{dataset_name}')

        for model_name in model_names:
            results = []

            if model_name.startswith('t5'):
                model_name_dir = '-'.join(model_name_map[model_name].split('-')[:2]).lower()
            elif model_name.startswith('f') or model_name.startswith('gpt'):
                model_name_dir = '-'.join(model_name_map[model_name].split('-')[:3]).lower()
            elif model_name.startswith('l3'):
                model_name_dir = 'llama3-instruct'
            elif model_name.startswith('l'):
                model_name_dir = '-'.join(model_name_map[model_name].split('-')[:4]).lower().replace('-hf', '').replace(
                    '-2', '')
            else:
                raise Exception(f'There is no model with the name: {model_name}')

            ablation_str = f'Syn{use_synonyms}_Ant{use_antonyms}_Hyper{use_hypernyms}_Hypo{use_hyponyms}'

            for transfer_direction in transfer_directions:
                for dataset_type in dataset_types:
                    directory_name = f'../predictions/{st_type}/{dataset_name}/{model_name_dir}'
                    primary_file_name = f'{model_name_map[model_name]}_{dataset_type}'
                    file_name = f'{primary_file_name}' if not ablation else f'{primary_file_name}_{ablation_str}'

                    predictions = pd.read_csv(f'{directory_name}/{file_name}.txt',
                                              sep='\t', encoding='utf-8', engine='python').dropna()

                    originals = predictions['Original'].values.tolist() if st_type in ['parallel', 'ablation'] else None
                    gt_sentences = predictions['Ground Truth'].values.tolist()
                    pred_sentences = predictions['Generated'].values.tolist()
                    # pred_sentences = predictions['Generated'].values.astype(str).tolist()

                    print(f'Dataset: {dataset_name} ({dataset_type})\n'
                          f'Model: {model_name}\n'
                          f'Transfer direction: {transfer_direction}\n'
                          f'Style transfer type: {st_type}')

                    r_bleu_scores = calculate_bleu(gts=gt_sentences, preds=pred_sentences)
                    r_bleu_score = round(r_bleu_scores['bleu'], 4)
                    print(f'rBLEU: {r_bleu_score}')

                    if originals is not None:
                        s_bleu_scores = calculate_bleu(gts=originals, preds=pred_sentences)
                        s_bleu_score = round(s_bleu_scores['bleu'], 4)
                        print(f'sBLEU: {s_bleu_score}')
                    else:
                        s_bleu_score = 0

                    r_bert_scores = calculate_bertscore(gts=gt_sentences, preds=pred_sentences)
                    r_bert_score = round(np.array(r_bert_scores['f1']).mean(), 4)
                    print(f'rBERTScore: {r_bert_score}')

                    if originals is not None:
                        s_bert_scores = calculate_bertscore(gts=originals, preds=pred_sentences)
                        s_bert_score = round(np.array(s_bert_scores['f1']).mean(), 4)
                        print(f'sBERTScore: {s_bert_score}')
                    else:
                        s_bert_score = 0

                    dataset_name_clean = dataset_name.split('_')[0]

                    accuracy_scores = calculate_accuracy(preds=pred_sentences,
                                                         style=int(transfer_direction[-1]) - 1,
                                                         model_name=f'DistilRoBERTa_{dataset_name_clean}',
                                                         tokenizer_name='distilroberta-base',
                                                         batch_size=batch_size)
                    accuracy_score = round(accuracy_scores['accuracy'], 4)
                    print(f'Accuracy: {accuracy_score}')

                    perplexity_scores = calculate_perplexity(preds=pred_sentences, batch_size=batch_size)
                    perplexity_score = round(perplexity_scores['mean_perplexity'], 4)
                    print(f'Perplexity: {perplexity_score}\n')

                    results.append([transfer_direction, dataset_type,
                                    r_bleu_score, s_bleu_score,
                                    r_bert_score, s_bert_score,
                                    accuracy_score,
                                    perplexity_score])

            results_df = pd.DataFrame(results)
            results_df.columns = ['Transfer Direction', 'Dataset Type',
                                  'rBLEU', 'sBLEU',
                                  'rBERTScore', 'sBERTScore',
                                  'Accuracy',
                                  'Perplexity']

            res_file_name = f'{model_name_map[model_name]}' if not ablation else f'{model_name_map[model_name]}_{ablation_str}'

            results_df.to_csv(f'../results/{st_type}/{dataset_name}/{res_file_name}_scores.txt',
                              sep='\t', index=False)
