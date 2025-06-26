import torch
import numpy as np
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification

models_root_location = '../models'


def calculate_bleu(gts, preds):
    """
    Compute the BLEU evaluation metric for the given predicted and ground
    truth sentences.
    :param gts: ground truth sentences
    :type gts: list(str)
    :param preds: predicted sentences
    :type preds: list(str)
    :return: evaluation scores
    :rtype: dict
    """

    metric = load('bleu')
    results = metric.compute(predictions=preds, references=gts)
    return results


def calculate_meteor(gts, preds):
    """
    Compute the METEOR evaluation metric for the given predicted and ground
    truth sentences.
    :param gts: ground truth sentences
    :type gts: list(str)
    :param preds: predicted sentences
    :type preds: list(str)
    :return: evaluation scores
    :rtype: dict
    """

    metric = load('meteor')
    results = metric.compute(predictions=preds, references=gts)
    return results


def calculate_rouge(gts, preds):
    """
    Compute the ROUGE-L evaluation metric for the given predicted and ground
    truth sentences.
    :param gts: ground truth sentences
    :type gts: list(str)
    :param preds: predicted sentences
    :type preds: list(str)
    :return: evaluation scores
    :rtype: dict
    """

    metric = load('rouge')
    results = metric.compute(predictions=preds, references=gts,
                             use_aggregator=True, rouge_types=['rougeL'])
    return results


def calculate_bertscore(gts, preds, batch_size=16):
    """
    Compute the BERTScore evaluation metric for the given predicted and ground
    truth sentences.
    :param gts: ground truth sentences
    :type gts: list(str)
    :param preds: predicted sentences
    :type preds: list(str)
    :param batch_size: batch size
    :type batch_size: int
    :return: evaluation scores
    :rtype: dict
    """

    metric = load('bertscore')
    results = metric.compute(predictions=preds, references=gts,
                             lang='en', model_type='distilroberta-base',
                             batch_size=batch_size, use_fast_tokenizer=True)
    return results


def calculate_accuracy(preds, style, model_name, tokenizer_name, batch_size=16):
    """
    Compute the accuracy evaluation metric with the DistilRoBERTa model for the
    given predicted sentences.
    :param preds: predicted sentences
    :type preds: list(str)
    :param style: target style (0 or 1)
    :type style: int
    :param model_name: model name
    :type model_name: str
    :param tokenizer_name: tokenizer name
    :type tokenizer_name: str
    :param batch_size: batch size
    :type batch_size: int
    :return: evaluation scores
    :rtype: dict
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(preds, truncation=True, padding=True)
    preds_input_ids = torch.from_numpy(np.array(encodings['input_ids']))
    preds_attention_masks = torch.from_numpy(np.array(encodings['attention_mask']))

    model = AutoModelForSequenceClassification.from_pretrained(f'{models_root_location}/{model_name}/',
                                                               num_labels=2)

    total, num_correct = 0, 0

    for i in range(0, len(preds), batch_size):
        input_ids = preds_input_ids[i:i + batch_size]
        attention_masks = preds_attention_masks[i:i + batch_size]

        logits = model(input_ids, attention_mask=attention_masks).logits
        labels = np.argmax(logits.detach().cpu().numpy(), axis=1)

        total += len(labels)
        num_correct += sum([label == style for label in labels])

    return {'total': total, 'num_correct': num_correct,
            'accuracy': num_correct * 1.0 / total}


def calculate_perplexity(preds, batch_size=16):
    """
    Compute the Perplexity evaluation metric for the given predicted sentences.
    :param preds: predicted sentences
    :type preds: list(str)
    :param batch_size: batch size
    :type batch_size: int
    :return: evaluation scores
    :rtype: dict
    """

    metric = load('perplexity', module_type='metric')
    results = metric.compute(predictions=preds, model_id='gpt2',
                             batch_size=batch_size, max_length=1024)
    return results


def calculate_all_scores(gts, preds, style, model_name, tokenizer_name, batch_size=16):
    """
    Compute all evaluation metrics for the given predicted sentences.
    :param gts: ground truth sentences
    :type gts: list(str)
    :param preds: predicted sentences
    :type preds: list(str)
    :param style: target style (0 or 1)
    :type style: int
    :param model_name: model name
    :type model_name: str
    :param tokenizer_name: tokenizer name
    :type tokenizer_name: str
    :param batch_size: batch size
    :type batch_size: int
    :return: evaluation scores
    :rtype: dict
    """

    scores = dict()

    bleu_scores = calculate_bleu(gts=gts, preds=preds)
    scores['BLEU'] = round(bleu_scores['bleu'], 8)

    meteor_scores = calculate_meteor(gts=gts, preds=preds)
    scores['METEOR'] = round(meteor_scores['meteor'], 8)

    rouge_scores = calculate_rouge(gts=gts, preds=preds)
    scores['ROUGE-L'] = round(rouge_scores['rougeL'], 8)

    bert_scores = calculate_bertscore(gts=gts, preds=preds,
                                      batch_size=batch_size)
    scores['BERTScore (precision)'] = round(np.array(bert_scores['precision']).mean(), 8)
    scores['BERTScore (recall)'] = round(np.array(bert_scores['recall']).mean(), 8)
    scores['BERTScore (F1)'] = round(np.array(bert_scores['f1']).mean(), 8)

    accuracy_scores = calculate_accuracy(preds=preds,
                                         style=style,
                                         model_name=model_name,
                                         tokenizer_name=tokenizer_name,
                                         batch_size=batch_size)
    scores['Accuracy'] = round(accuracy_scores['accuracy'], 8)

    perplexity_scores = calculate_perplexity(preds=preds, batch_size=batch_size)
    scores['Perplexity'] = round(perplexity_scores['mean_perplexity'], 8)

    return scores
