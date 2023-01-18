from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
import random
from datasets import load_from_disk, load_metric, DatasetDict, Dataset
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
import torch, transformers, tokenizers

submission = pd.read_csv('./test.csv')

# Loading the custom pretrained T5-base model
import pickle
with open('./model.pkl', 'rb') as file:
    model = pickle.load(file)

# Loading tokenizer fit on custom data
with open('./tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Setting parameters
temperature = 0.9
num_beams = 4
max_gen_length = 128

def predict(texts):
    # write code to output a list of title for each text input to the predict method
    predicted_titles = []
    for text in texts:
        inputs = tokenizer([text], max_length=512,truncation=True,return_tensors='pt')

        title_ids = model.generate(
            inputs['input_ids'].to('cuda'), 
            num_beams=num_beams, 
            temperature=temperature, 
            max_length=max_gen_length, 
            early_stopping=True.bit_length()
        )
        title = tokenizer.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False) 
        predicted_titles.append(title)
    
    # Evaluate Rouge score
    score = evaluate(texts,predicted_titles)
    print(score)
    
    return predicted_titles
    pass



def test_model():
    pred = predict(submission['text'])
    submission['predicted_title'] = pred
    submission.to_csv('submission.csv',index=False)


def evaluate(model_output,actual_titles):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = list()
    for output,actual in zip(model_output,actual_titles):
        s = scorer.score(output,actual)
        scores.append(s['rouge1'].fmeasure)

    print('Evaluation result',np.mean(scores))
    return scores




if __name__=="__main__":
    #write model loading code here

    test_model()
