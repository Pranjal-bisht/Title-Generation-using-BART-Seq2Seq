# Import Libraries
import numpy as np
import pandas as pd
import random
from datasets import load_metric, DatasetDict, Dataset

import re
import nltk
from nltk.tokenize import sent_tokenize

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer


# Lowercase, remove punctuation and numbers from kernel titles
def clean_title(title):
    '''
    Function to lowercase, remove punctuation and numbers from kernel titles
    '''
    # lowercase
    title = str(title).lower()
    # replace punctuation into spaces
    title = re.sub(r"[,.;@#?!&$%<>-_*/\()~='+:`]+\ *", " ", title)
    title = re.sub('-', ' ', title)
    title = re.sub("''", ' ', title)
    # replace numbers into spaces
    title = re.sub(r"[0123456789]+\ *", " ", title)
    #remove duplicated spaces
    title = re.sub(' +', ' ', title)
    
    return title.strip()

MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128


# Initialize T5-base tokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-base')

# Tokenization of text data
def preprocess_data(example):
    
    model_inputs = tokenizer(list(example['input_text']), max_length=MAX_SOURCE_LEN, padding=True, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(list(example['target_text']), max_length=MAX_TARGET_LEN, padding=True, truncation=True)

    # Replace all pad token ids in the labels by -100 to ignore padding in the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs['labels'] = labels["input_ids"]

    return model_inputs



# Define ROGUE metrics on evaluation data
metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Compute ROUGE scores and get the median scores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return {k: round(v, 4) for k, v in result.items()}




def train():
    # Read the data
    file_path = './train.csv'
    df = pd.read_csv(file_path)
    new_df = df.applymap(clean_title)

    # Data cleaning and Preprocessing
    df = df[['title','text']]
    df.columns = ['target_text','input_text']
    df = df.dropna()

    # creating a list of columns
    swap_list = ["input_text","target_text"]

    # Swapping the columns
    df = df.reindex(columns=swap_list)
    df=df.reset_index() 
    df=df.drop(['index'],axis=1)

    # Splitting the dataset
    ds = DatasetDict() # making Dataframe to datasetDict for easier processing while tokenization
    ds['train'] = Dataset.from_pandas(df[:10001])
    ds['validation'] = Dataset.from_pandas(df[10001:])
    

    # Apply preprocess_data() to the whole dataset
    processed_dataset = ds.map(
        preprocess_data,
        batched=True,
        remove_columns=['input_text', 'target_text'],
        desc="Running tokenizer on dataset",
    )

    # Initialize T5-base model
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base') 

    # Dynamic padding in batch using a data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  


    # Defining Training parameters
    batch_size = 8
    num_epochs = 5
    learning_rate = 5.6e-5
    weight_decay = 0.01
    log_every = 50
    eval_every = 1000
    lr_scheduler_type = "linear"

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="model-t5-base",
        evaluation_strategy="steps",
        eval_steps=eval_every,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_steps=500,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        logging_steps=log_every,
        group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        report_to="wandb",
        resume_from_checkpoint=True,
    )

    # Define the trainer
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    ) 

    # Fine-tune the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate(eval_dataset=processed_dataset["validation"])

    pass
    
    
if __name__=="__main__":
    print('Running training script')
    train()