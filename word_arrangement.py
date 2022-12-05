from transformers import T5ForConditionalGeneration,\
    Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from datasets import load_dataset
import nltk
import os
import wandb
from nltk import SnowballStemmer
from utils import tokenizer, model,device, stem_mapping, tokenize_function,compute_metrics, inference
nltk.download('punkt')

def prepare_input_output_WordArrange(list_of_words, sentence):
    # Given:'[ski, mountain, skier]', "Skier skis down the mountain"
    # The function returns input and output of the format:('[ski, mountain, skier]', '[skier, ski, mountain]')
    # Where the input words order is the original order
    # and the output words order is the order they appeared in the sentence. The "correct" order.
    stemmer = SnowballStemmer('english')
    sentence = sentence.replace(".", "")
    list_sentence = sentence.split()
    list_stem_sentence = []
    for w in list_sentence:
        if stemmer.stem(w) in stem_mapping:
            stem_word = stem_mapping[stemmer.stem(w)]
        else:
            stem_word = stemmer.stem(w)
        list_stem_sentence.append(stem_word)
    output_list="["
    for word_S in list_stem_sentence:
        if (word_S in list_of_words):
            output_list+=(word_S+", ")
    input_list="["
    for word_l in list_of_words:
        if(word_l in output_list):
            input_list+=(word_l+", ")
    output_list=output_list[: -2]
    output_list+="]"
    input_list=input_list[: -2]
    input_list+= "]"
    return input_list, output_list


def prepare_data_WordArrange(row):
    # prepare data for fine-tune
    list_of_words= row["concepts"]
    sentence = row["target"]
    input_format, output_format = prepare_input_output_WordArrange(list_of_words, sentence)
    return {'input_format': input_format, 'label_format': output_format}


def fine_tune_for_word_arrangement():
    wandb.init(project="word_arrangement")
    datasetsCG = load_dataset("common_gen")

    datasetsCG_p = datasetsCG.map(prepare_data_WordArrange)
    tokenized_datasets = datasetsCG_p.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    batch_size = 64
    training_args = Seq2SeqTrainingArguments(
        "trainer-WA",
        report_to="wandb",
        evaluation_strategy="epoch",
        predict_with_generate=True,
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        per_gpu_train_batch_size=batch_size,
        per_gpu_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=7,
        fp16=False,
        seed=123
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    checkpoints_folder_name = "word_arrangement"
    full_path = "/data/home/gilatt/CommonGen/checkpoints/{}".format(checkpoints_folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    trainer.save_model(full_path)
