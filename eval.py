import pickle
import os
import re
import collections
import sys
import numpy as np
import nltk
from tqdm import tqdm
import spacy
from collections import Counter

sys.path.append('./pyeval')
from pyeval.bleu.bleu import Bleu
from pyeval.rouge.rouge import Rouge
from pyeval.meteor.meteor import Meteor
from pyeval.distinct.distinct import Distinct

class Evaluate(object):
    def __init__(self):
        self.metrics = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.metrics:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, candidate_list, reference_list):
        # make dictionary
        hypo = {}
        ref = {}
        for i in range(len(candidate_list)):
            hypo[i] = [candidate_list[i]]
            ref[i] = [reference_list[i]]

        # compute scores
        final_scores = self.score(ref, hypo)

        return final_scores

def eval_acc(predictions, raw, cache=None, compute_cache=False):
    def lower(text):
        if isinstance(text, str):
            text = text.strip().lower()
            text = ' '.join(nltk.word_tokenize(text))
            return text.strip()
        return [lower(item) for item in text]
    nlp = spacy.load("en_core_web_sm")
    ent_f1 = []
    k_f1 = []
    sent_acc = []
    build_cache = []
    if cache is not None:
        raw = cache
    if compute_cache:
        predictions = tqdm(raw)

    for pred, example in zip(predictions, raw):
        if cache is not None:
            label_knowledge, label_response, label_ents, all_candidates = example
        else:
            if isinstance(example['title'], list):
                label_knowledge = [lower(f'{t} {s}') for t, s in zip(example['title'], example['checked_sentence'])]
            else:
                label_knowledge = [lower(example['title'] + ' ' + example['checked_sentence'])]
            label_response = lower(example['labels'][0])
            label_ents = [ent.text for ent in nlp(label_response).ents]
            all_candidates = [lower(f'{title} {sentence}') for title in example['knowledge'] for sentence in
                              example['knowledge'][title]]
        if compute_cache:
            build_cache.append([label_knowledge, label_response, label_ents, all_candidates])
        else:
            pred_response = lower(pred)
            pred_ents = [ent.text for ent in nlp(pred_response).ents]
            if len(label_ents) > 0:
                ent_f1.append(f1_score(' '.join(pred_ents), [' '.join(label_ents)]))
            if len(label_knowledge) == 0:
                k_f1.append(0)
            else:
                k_f1.append(f1_score(pred_response, label_knowledge))
            max_candidates_f1 = max([f1_score(sent, [pred_response]) for sent in all_candidates])
            sent_acc.append(int(max_candidates_f1 == k_f1[-1]))
    if compute_cache:
        return build_cache
    return {'KF1': sum(k_f1) / len(k_f1) * 100,
            'EntF1': sum(ent_f1) / len(ent_f1) * 100,
            'ACC': sum(sent_acc) / len(sent_acc) * 100}

# ==================================================================================================
# F1 Score
# ==================================================================================================
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def f1_score(guess, answers):
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)

