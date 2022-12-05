from transformers import T5TokenizerFast, T5ForConditionalGeneration,\
     Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
from datasets import load_dataset
import nltk
import wandb
import os
from nltk import SnowballStemmer
nltk.download('punkt')
from utils import tokenizer, model, device,stem_mapping, tokenize_function,compute_metrics, inference


def make_masked_input(list_of_words, sentence):
    # given : [ "wag", "tail", "dog" ],"The dog is wagging his tail."
    # return : "<extra_id_0> dog <extra_id_1> wag <extra_id_2> tail<extra_id_3>"
     stemmer = SnowballStemmer('english')
     sentence = sentence.replace(".", "")
     input_list = ["<extra_id_0>"]
     special_tok = "<extra_id_0>"
     index_input = 1
     list_sentence = sentence.split()
     list_stem_sentence = []
     for w in list_sentence:
         if stemmer.stem(w) in stem_mapping:
             stem_word = stem_mapping[stemmer.stem(w)]
         else:
             stem_word = stemmer.stem(w)
         list_stem_sentence.append(stem_word)
     for word_S in list_stem_sentence:
          if (word_S in list_of_words):
               input_list.append(word_S)
               input_list.append(special_tok.replace("0", str(index_input)))
               index_input += 1
     input_format = " ".join(input_list)
     return input_format


def prepare_data_SentenceGen(row):
    # prepare data for fine-tune
    list_of_words= row["concepts"]
    sentence = row["target"]
    input_format = make_masked_input(list_of_words, sentence)
    return {'input_format': input_format, 'label_format': sentence}



def fine_tune_for_sentence_generator():
    wandb.init(project="sentence_generator")
    datasetsCG = load_dataset("common_gen")

    datasetsCG_p = datasetsCG.map(prepare_data_SentenceGen)
    tokenized_datasets = datasetsCG_p.map(tokenize_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    batch_size = 64
    training_args = Seq2SeqTrainingArguments(
        "trainer-SG",
        report_to="wandb",
        evaluation_strategy="epoch",
        predict_with_generate=True,
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        per_gpu_train_batch_size=batch_size,
        per_gpu_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=9,
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

    checkpoints_folder_name = "sentence_generator"
    full_path = "/data/home/gilatt/CommonGen/checkpoints/{}".format(checkpoints_folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    trainer.save_model(full_path)

