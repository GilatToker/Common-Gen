from word_arrangement import fine_tune_for_word_arrangement
from sentence_generator import fine_tune_for_sentence_generator
from utils import overlapping_words, count_overlapping_words, inference,  remove_pad_S, evaluate_func
import torch
import os
from transformers import T5ForConditionalGeneration
from datasets import load_dataset
import pandas as pd


def preper_input_from_wordA_to_sentenceG(output_of_word_arrangement):
    # Given: ['<pad> [skier, ski, mountain]</s>']
    # Return: "<extra_id_0> skier <extra_id_1> ski <extra_id_2> mountain <extra_id_3>"

    # initializing substrings
    string = output_of_word_arrangement[0]
    sub1 = "["
    sub2 = "]"
    # getting index of substrings
    if sub1 in string and sub2 in string:
        idx1 = string.index(sub1)
        idx2 = string.index(sub2)
        # length of substring 1 is added to
        # get string from next character
        res = string[(idx1-1) + len(sub1) + 1: idx2]
        list_of_words = res.split(",")
    else:
        list_of_words = ["error","error","error"]
    input_list = ["<extra_id_0>"]
    special_tok = "<extra_id_0>"
    index_input = 1
    for word in list_of_words:
        input_list.append(word)
        input_list.append(special_tok.replace("0", str(index_input)))
        index_input += 1
    input_format = " ".join(input_list)
    return input_format

def fine_tune_models():
    fine_tune_for_word_arrangement()
    fine_tune_for_sentence_generator()

def run_my_words(word1, word2, word3,beam_or_sample, beam_size, temp, top_k, top_p):
    input_for_word_arrangement = "[" +word1+ ", "+word2 + ", "+ word3 + "]"
    output_of_word_arrangement = inference(input_for_word_arrangement, word_arrangement_model,
                                           beam_or_sample, beam_size, temp, top_k, top_p)
    input_for_sentence_generator = preper_input_from_wordA_to_sentenceG(output_of_word_arrangement)
    model_output = inference(input_for_sentence_generator, sentence_generator_model,
                             beam_or_sample, beam_size, temp, top_k, top_p)
    return model_output


def run_validation_by_metrics(datasetsCG, size, beam_or_sample, beam_size, temp, top_k, top_p):
    # Given generate function args create df row with all the metric scores
    prediction = []
    references = []
    for i in range(size):
        list_of_words = datasetsCG["validation"][i]['concepts']
        model_output = run_my_words(list_of_words[0], list_of_words[1], list_of_words[2], beam_or_sample, beam_size, temp, top_k, top_p)
        model_string = remove_pad_S(model_output)
        prediction.append(model_string)
        validation = datasetsCG["validation"][i]['target']
        references.append(validation)
    metric_scores = evaluate_func(prediction, references)
    overlapping = overlapping_words(references, prediction)
    new_row = {'parameter_value':f'sample or beam: {beam_or_sample},beam_size={beam_size},temp={temp},top_k = {top_k},top_p={top_p}',
               'gleu_1': metric_scores["gleu_1"],
               'glue_2': metric_scores["gleu_2"], 'glue_4': metric_scores["gleu_4"], 'rouge1_p': metric_scores["rouge1_p"],
               'rouge1_r': metric_scores["rouge1_r"], 'rouge1_f': metric_scores["rouge1_f"],
               'rouge2_p': metric_scores["rouge2_p"], 'rouge2_r': metric_scores["rouge2_r"],
               'rouge2_f': metric_scores["rouge2_f"], 'rougeL_p': metric_scores["rougeL_p"], 'rougeL_r': metric_scores["rougeL_r"],
               'rougeL_f': metric_scores["rougeL_f"], 'sacrebleu': metric_scores["sacrebleu"],
               'overlapping_words': overlapping}
    return new_row

def create_df_with_diff_args(datasetsCG):
    # test different generate function args to decide which one is best.
    datasetsCG["validation"] = load_dataset("common_gen", split="validation[:25%]")
    size = datasetsCG["validation"].num_rows
    df = pd.DataFrame(columns=['parameter_value', 'gleu_1', 'glue_2', 'glue_4',
                               'rouge1_p', 'rouge1_r', 'rouge1_f', 'rouge2_p', 'rouge2_r', 'rouge2_f',
                               'rougeL_p', 'rougeL_r', 'rougeL_f', 'sacrebleu', 'overlapping_words'])

    #beam method
    for beam_size in range(3,10):
        row = run_validation_by_metrics(datasetsCG,size, "beam", beam_size, 0, 0, 0)
        df = df.append(row, ignore_index=True)
    #sample method
    for temp in [0.5,0.6,0.7,0.8,0.9, 1]:
        for top_k in range(40,100,10):
            row = run_validation_by_metrics(datasetsCG,size,"sample", 0, temp, top_k, 0)
            df = df.append(row, ignore_index=True)
        for top_p in [0.9, 0.92, 0.95, 0.97, 0.99, 0.999]:
            row = run_validation_by_metrics(datasetsCG,size, "sample", 0, temp, 0,top_p)
            df = df.append(row, ignore_index=True)
    return df

def print_sentences():
    all_lists =[
        # TEST SET
        ["couple", "table", "sit"],
        ["run", "field", "team"],
        ["demonstrate", "machine", "use"],
        ["push", "mow", "lawn"],
        ["drive", "road", "car"],
        ["continue", "smoke", "cigarette"],
        ["score", "win", "game"],
        ["meat", "cut", "knife"],
        ["read", "lay", "bed"],
        ["swimmer", "pool", "compete"],
        # MU OWN WORDS
        ["work", "why","hard"],
        ["robot","best","doctor"],
        ["come", "run", "you"],
        ["time", "spend", "family"],
        ["London", "entertainment", "friends"],
        ["dream", "scary", "yesterday"],
        ["grandmother", "me", "present"],
        ["application","NLP","new"],
        ["come","party","girl"],
        ["friend","cry","hurt"],
        ["go","love","you"],
        ["forest","run","man"]
    ]
    for list in all_lists:
        print(list)

        # chosen: beam, 5
        model_output = run_my_words(list[0], list[1], list[2], "beam", 5, 0, 0, 0)
        print(remove_pad_S(model_output))


if __name__ == '__main__':
    datasetsCG = load_dataset("common_gen")
    # fine_tune_models()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoints_folder_name_WA = "word_arrangement"
    word_arrangement_model = T5ForConditionalGeneration.from_pretrained(
        "/data/home/gilatt/CommonGen/checkpoints/{}".format(checkpoints_folder_name_WA))
    word_arrangement_model = word_arrangement_model.to(device)
    checkpoints_folder_name_SG = "sentence_generator"
    sentence_generator_model=T5ForConditionalGeneration.from_pretrained(
        "/data/home/gilatt/CommonGen/checkpoints/{}".format(checkpoints_folder_name_SG))
    sentence_generator_model = sentence_generator_model.to(device)
    print_sentences()