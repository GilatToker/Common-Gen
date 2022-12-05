import evaluate
import numpy as np
import nltk
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from nltk import SnowballStemmer
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_metric
from typing import List, Tuple, Dict, Any, Union, Callable, Optional
from statistics import mean


# define tokenizer and model
checkpoint = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)


def stem_mapping_function():
    # Using stemming: the process of reducing a word to its word stem
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    stemmer = SnowballStemmer('english')
    words = {w for w in tokenizer.vocab if w.startswith('▁') and len(w) > 1}
    stemmed_words = defaultdict(list)
    for w in words:
        w = w.replace('▁', '')
        stemmed_word = stemmer.stem(w)
        stemmed_words[stemmed_word].append(w)
    stem_mapping = {}
    for stemmed_word, word_list in stemmed_words.items():
        word_list = sorted(word_list, key=lambda x: len(x))
        lowers = [v for v in word_list if v[0].islower()]
        if len(lowers) > 0:
            rep_word = lowers[0]
        else:
            rep_word = word_list[0]
        stem_mapping[stemmed_word] = rep_word
    return stem_mapping

stem_mapping = stem_mapping_function()

def tokenize_function(row):
    model_inputs = tokenizer(row["input_format"], truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(row["label_format"], truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    metric = evaluate.load("sacrebleu")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #  newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result

def count_overlapping_words(test_sentence,generated_sentence):
    # another metric to evaluate the model: Given 2 sentences returns the amount of overlapping stemmed words
    stemmer = SnowballStemmer('english')
    test_sentence = test_sentence.replace(".", "")
    test_words = test_sentence.split()
    list_test_stem_words = []
    for w in test_words:
        if stemmer.stem(w) in stem_mapping:
            stem_word = stem_mapping[stemmer.stem(w)]
        else:
            stem_word = stemmer.stem(w)
        list_test_stem_words.append(stem_word)

    generated_sentence = generated_sentence.replace(".", "")
    generated_words = generated_sentence.split()
    list_generated_stem_words = []
    for w in generated_words:
        if stemmer.stem(w) in stem_mapping:
            stem_word = stem_mapping[stemmer.stem(w)]
        else:
            stem_word = stemmer.stem(w)
        list_generated_stem_words.append(stem_word)

    over_lapping_words = 0
    total_count =0
    for word in list_generated_stem_words:
        total_count+=1
        if word in list_test_stem_words:
            over_lapping_words+=1
            list_test_stem_words.remove(word)
    return (over_lapping_words/total_count)*100

def overlapping_words(test_sentence_list,generated_sentence_list):
    # return the mean overlapping score for all sentences
    over_lapping_words_results =[]
    for i in range(len(test_sentence_list)):
        over_lapping_words_results.append(count_overlapping_words(test_sentence_list[i], generated_sentence_list[i]))
    return mean(over_lapping_words_results)


def remove_pad_S(list):
    #from : ['<pad> A man looks at a giraffe standing in a field.</s>']
    # to: 'A man looks at a giraffe standing in a field.'
    string=list[0]
    sub1 = "<pad> "
    sub2 = "</s>"
    if (sub1 in string and sub2 in string):
        # getting index of substrings
        idx1 = string.index(sub1)
        idx2 = string.index(sub2)
        # length of substring 1 is added to
        # get string from next character
        res = string[(idx1 - 1) + len(sub1) + 1: idx2]
    elif (sub1 in string and sub2 not in string):
        idx1 = string.index(sub1)
        res = string[(idx1 - 1) + len(sub1) + 1:]
    elif (sub1 not in string and sub2 in string):
        idx2 = string.index(sub2)
        res = string[:idx2]
    else:
        res = string
    return res



def inference(sentence, moedl_I, beam_or_sample, beam_size, temp, top_k, top_p):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    if (beam_or_sample=="beam"):
        outputs = moedl_I.generate(input_ids,
                                   max_length=20,
                                   num_beams= beam_size,
                                   no_repeat_ngram_size=4,
                                   early_stopping=True)
    else:
        outputs = moedl_I.generate(input_ids,
                                   max_length=20,
                                   do_sample=True,
                                   top_k=top_k,
                                   temperature= temp,
                                   top_p=top_p,
                                   early_stopping=True)
    sequences = tokenizer.batch_decode(outputs)
    return sequences

def load_metrics(metrics: Optional[Union[List[str], str]] = None) -> Dict[str, Callable]:
    possible_metrics = ['gleu', 'rouge', 'sacrebleu', 'bertscore']
    if 'all' in metrics:
        metrics = possible_metrics
    else:
        metrics = []
    metric_funcs = {}
    for metric in metrics:
        if metric == 'gleu':
            hf_gleu = load_metric('google_bleu')
            def gleu(predictions, references):
                # לוודא באיזה תבנית צריך להיות predictions, references
                predictions = [t.split() for t in predictions]
                references = [[t.split()] for t in references]
                score_dict = {}
                for i in [1, 2, 4]:
                    score_dict[f'gleu_{i}'] = hf_gleu.compute(predictions=predictions,
                                                              references=references, max_len=i)['google_bleu'] * 100
                return score_dict

            metric_funcs[metric] = gleu
        elif metric == 'rouge':
            hf_rouge = load_metric('rouge')

            def rouge(predictions, references):
                rouge_scores = hf_rouge.compute(predictions=predictions, references=references)
                score_dict = {}
                for k in ['rouge1', 'rouge2', 'rougeL']:
                    score_dict.update({f'{k}_p': rouge_scores[k].mid.precision * 100,
                                       f'{k}_r': rouge_scores[k].mid.recall * 100,
                                       f'{k}_f': rouge_scores[k].mid.fmeasure * 100})
                return score_dict

            metric_funcs[metric] = rouge

        elif metric == 'sacrebleu':
            hf_sacrebleu = load_metric('sacrebleu')

            def sacrebleu(predictions, references):
                return {'sacrebleu': hf_sacrebleu.compute(predictions=predictions,
                                                          references=[[r] for r in references])['score']}
            metric_funcs[metric] = sacrebleu

    return metric_funcs

def evaluate_func(prediction: List[str],references: List[str])-> Dict[str,float]:
    result = {}
    metric_funcs = load_metrics("all")
    for metric, metric_func in metric_funcs.items():
        result.update(metric_func(prediction, references))
    return {k: round(v, 2) for k, v in result.items()}


