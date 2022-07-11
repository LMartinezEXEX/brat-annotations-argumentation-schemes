import glob
from sklearn.metrics import f1_score
import sys
from datasets import Dataset
import torch
from tqdm import tqdm
from sklearn import metrics
import argparse
from pysentimiento import preprocessing
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument('components', type=str, nargs='+', help="Name of the component that wants to be identified")
parser.add_argument('--solver', type=str, default="liblinear", help="Name of the solver to be used in the Logistic regression")
parser.add_argument('--c', type=float, default=1.0, help="Inverse of regularization strength")

args = parser.parse_args()

REP = 0
NUMBER_OF_PARTITIONS = 10
C = args.c
SOLVER = args.solver
components = args.components
component = components[0]


def delete_unwanted_chars(text):
        return text.replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")

def labelAllExamples(filePatterns, component, multidataset = False, add_annotator_info = False):
    all_tweets = []
    all_labels = []
    if multidataset:
        datasets = []
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
             # TODO: sacar todos los caracteres especiales
            tweet_text = delete_unwanted_chars(tweet.read())
            is_not_argumentative = False
            type_of_premise = -1
            component_text = []
            premise_codes = []
            for idx, word in enumerate(annotations):
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    if current_component.startswith("NonArgumentative"):
                        is_not_argumentative = True
                        break
                    if current_component.startswith(component):
                       component_text.append(delete_unwanted_chars(ann[2]))
                       premise_codes.append(ann[0])
                    if current_component.startswith("QuadrantType"):
                       info_splitted = current_component.split(" ")
                       if info_splitted[1].strip() in premise_codes:
                           type_of_premise_text = info_splitted[2].strip()
                           if type_of_premise_text == "fact":
                               type_of_premise = 0
                           elif type_of_premise_text == "value":
                               type_of_premise = 1
                           elif type_of_premise_text == "policy":
                               type_of_premise = 2
            if not is_not_argumentative:
                if add_annotator_info:
                    tweet_text = tweet_text + " " + component + ": " + " ".join(component_text)
                preprocessed_text = preprocessing.preprocess_tweet(tweet_text)
                if multidataset:
                    dicc = {"text": [preprocessed_text], "label": [is_not_argumentative]}
                    datasets.append([Dataset.from_dict(dicc), preprocessed_text])
                else:
                    all_tweets.append(preprocessed_text)
                    all_labels.append(type_of_premise)
                    print(preprocessed_text)
                    print(type_of_premise)
                    print("---------------------------------------------------------------------------------------")
    if multidataset:
        return datasets

    return pd.DataFrame([all_tweets, all_labels], index=["text", "labels"]).T

#    def tokenize_and_align_labels_per_example(example):
#        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
#
#        labels = []
#        for i, label in enumerate(example[f"labels"]):
#            word_ids = tokenized_inputs.word_ids(batch_index=i)
#            previous_word_idx = None
#            label_ids = []
#            for word_idx in word_ids:  # Set the special tokens to -100.
#                if word_idx is None:
#                    label_ids.append(-100)
#                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                    label_ids.append(label[word_idx])
#                else:
#                    label_ids.append(-100)
#                previous_word_idx = word_idx
#            labels.append(label_ids)
#
#        tokenized_inputs["labels"] = labels
#        return tokenized_inputs
#
#    if is_multi:
#        return [{"dataset": data[0].map(tokenize_and_align_labels_per_example, batched=True), "text": data[1]} for data in dataset]
#    return dataset.map(tokenize_and_align_labels_per_example, batched=True)


def tokenize_examples(dataset, tokenizer, embeddings_model):

    def extract_embeddings(example):
        tokenized_inputs = tokenizer(example["text"], truncation=True, return_tensors="pt")
        output = embeddings_model(**tokenized_inputs)
        embeddings = output.last_hidden_state.squeeze()
        embeddings = torch.sum(embeddings, dim=0)
        return embeddings

    all_embeddings = dataset.apply(extract_embeddings, axis=1)
    return pd.DataFrame([all_embeddings, dataset["labels"]], index=["text", "labels"]).T

def train(model, embeddings_model, tokenizer, component, train_partition_patterns, random_state=0, with_embeddings = True):

    training_set = labelAllExamples(train_partition_patterns, component)
    if with_embeddings:
        training_set = tokenize_examples(training_set, tokenizer, embeddings_model).to_numpy()
        
        X = np.array([t.detach().numpy() for t in training_set[:,0]])
        y = training_set[:,1]
        y = y.astype('int')
        filter_cond = [True if x != -100 else False for x in y]
        y = y[filter_cond]
        X = X[filter_cond]

    else:
        X = training_set.drop("labels", axis=1)
        X.fillna(0, inplace=True)
        v = DictVectorizer(sparse=False)
        X = v.fit_transform(X.to_dict('records'))
    
        y = training_set.labels.values
        y = y.astype('int')



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, shuffle=True, random_state=random_state)

    logreg = LogisticRegression(C=C, solver=SOLVER, class_weight = "balanced")
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)


    filename = "results_test_{}_{}_LR_{}_type-of-premise_{}".format(C, SOLVER, REP, component)

    with open(filename, 'w') as w:
        w.write("{},{},{},{}".format(metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred, average="macro"), metrics.recall_score(y_test, y_pred, average="macro"), metrics.f1_score(y_test, y_pred, average="macro")))


#    print(y_pred)



#    print(list(zip(y_test, y_pred)))

#    print(X)


filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS + 1)]


for i in range(3):
    REP = REP + 1
    for cmponent in components:
        component = cmponent
        model = LogisticRegression()
        embeddings_model = AutoModel.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        train(model, embeddings_model, tokenizer, component, filePatterns, random_state=i, with_embeddings=True)


