import glob
from sklearn.metrics import f1_score
import sys
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
from tqdm import tqdm
from transformers import EvalPrediction
from sklearn import metrics
import argparse
from transformers import EarlyStoppingCallback
from pysentimiento import preprocessing

parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument('--modelname', type=str, default="roberta-base", help="Name of the language model to be downloaded from huggingface")
parser.add_argument('--lr', type=float, default=2e-05, help="Learning rate for training the model. Default value is 2e-05")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation. Default is 16")

args = parser.parse_args()


LEARNING_RATE = args.lr
NUMBER_OF_PARTITIONS = 10
device = "cuda"
EPOCHS = 10
BATCH_SIZE=args.batch_size
MODEL_NAME = args.modelname

def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids

    true_labels = [str(label) for label in labels]
    true_predictions = [str(pred) for pred in preds]

    print(true_labels)
    print(true_predictions)


    f1 = metrics.f1_score(true_labels, true_predictions, average="binary", pos_label='1')
    print("F1: {}".format(f1))

    acc = metrics.accuracy_score(true_labels, true_predictions)
    print("ACC: {}".format(acc))

    recall = metrics.recall_score(true_labels, true_predictions, average="binary", pos_label='1')
    print("Recall: {}".format(recall))

    precision = metrics.precision_score(true_labels, true_predictions, average="binary", pos_label='1')
    print("Precision: {}".format(precision))

    f1_micro = metrics.f1_score(true_labels, true_predictions, average="micro")
    print(f1_micro)

    recall_micro = metrics.recall_score(true_labels, true_predictions, average="micro")
    print(recall_micro)

    precision_micro = metrics.precision_score(true_labels, true_predictions, average="micro")
    print(precision_micro)


    w = open("./results_{}_{}_{}-metrics".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), "NonArgumentatives"), "a")

    w.write("{},{},{},{},{},{},{}\n".format(str(acc), str(f1), str(precision), str(recall), str(f1_micro), str(precision_micro), str(recall_micro)))
    w.close()

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def labelAllExamples(filePatterns, multidataset = False):
    all_tweets = []
    all_labels = []
    if multidataset:
        datasets = []
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
             # TODO: sacar todos los caracteres especiales
            tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "")
            is_not_argumentative = 0
            for idx, word in enumerate(annotations):
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    if current_component.startswith("NonArgumentative"):
                        is_not_argumentative = 1
                        break
         
            preprocessed_text = preprocessing.preprocess_tweet(tweet_text)
            if multidataset:
                dicc = {"text": [preprocessed_text], "label": [is_not_argumentative]}
                datasets.append([Dataset.from_dict(dicc), preprocessed_text])
            else:
                all_tweets.append(preprocessed_text)
                all_labels.append(is_not_argumentative)
    if multidataset:
        return datasets

    ans = {"text": all_tweets, "label": all_labels}
    return Dataset.from_dict(ans)


def tokenize_preprocess(dataset, tokenizer, is_multi = False):
    def tokenize_preprocess_per_example(example):
        return tokenizer(example["text"], truncation=True)

    if is_multi:
        return [{"dataset": data[0].map(tokenize_preprocess_per_example, batched=True), "text": data[1]} for data in dataset]

    return dataset.map(tokenize_preprocess_per_example, batched=True)


def train(epochs, model, tokenizer, train_partition_patterns, dev_partition_patterns, test_partition_patterns):

    training_set = tokenize_preprocess(labelAllExamples(train_partition_patterns), tokenizer)
    dev_set = tokenize_preprocess(labelAllExamples(dev_partition_patterns), tokenizer)
    test_set = tokenize_preprocess(labelAllExamples(test_partition_patterns), tokenizer)
    test_set_one_example = tokenize_preprocess(labelAllExamples(test_partition_patterns, multidataset = True), tokenizer, is_multi=True)
    
    training_args = TrainingArguments(
        output_dir="./results_eval_{}_{}_ARGUMENTATIVE".format(LEARNING_RATE, MODEL_NAME.replace("/", "-")),
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=5,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        report_to="none",
        metric_for_best_model='f1',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= compute_metrics_f1,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=7)]
    ) 

    trainer.train()
    print("--------------------------------------------------------------------------------------------------evaluation---------------------------------------------------------------------------------------------------------------------")
    print(trainer.evaluate())

    results = trainer.predict(test_set)
    filename = "./results_test_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), "NonArgumentative")
    with open(filename, "w") as writer:
        print(results.metrics)
        writer.write("{},{},{},{}".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"]))

    examples_filename = "./examples_test_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), "NonArgumentative")
    with open(examples_filename, "w") as writer:
        for dtset in test_set_one_example:
            result = trainer.predict(dtset["dataset"])
            writer.write("{}\t{}\t{}\n".format(dtset["text"], result.predictions.argmax(-1), result.label_ids))
    
        
        


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS+1)]
train(0, model, tokenizer, filePatterns[:8], filePatterns[8:9], filePatterns[9:])


