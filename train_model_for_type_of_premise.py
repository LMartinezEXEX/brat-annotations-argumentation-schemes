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
parser.add_argument('components', type=str, nargs='+', help="Name of the component that wants to be identified")
parser.add_argument('--modelname', type=str, default="roberta-base", help="Name of the language model to be downloaded from huggingface")
parser.add_argument('--lr', type=float, default=2e-05, help="Learning rate for training the model. Default value is 2e-05")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation. Default is 16")

args = parser.parse_args()


LEARNING_RATE = args.lr
NUMBER_OF_PARTITIONS = 10
device = "cuda"
EPOCHS = 20
BATCH_SIZE=args.batch_size
MODEL_NAME = args.modelname
type_of_premises = args.components
component = type_of_premises[0]

def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids

    true_labels = [str(label) for label in labels]
    true_predictions = [str(pred) for pred in preds]

    print(true_labels)
    print(true_predictions)


    f1 = metrics.f1_score(true_labels, true_predictions, average="macro")
    print("F1: {}".format(f1))

    acc = metrics.accuracy_score(true_labels, true_predictions)
    print("ACC: {}".format(acc))

    recall = metrics.recall_score(true_labels, true_predictions, average="macro")
    print("Recall: {}".format(recall))

    precision = metrics.precision_score(true_labels, true_predictions, average="macro")
    print("Precision: {}".format(precision))

    f1_micro = metrics.f1_score(true_labels, true_predictions, average="micro")
    print(f1_micro)

    recall_micro = metrics.recall_score(true_labels, true_predictions, average="micro")
    print(recall_micro)

    precision_micro = metrics.precision_score(true_labels, true_predictions, average="micro")
    print(precision_micro)


    w = open("./results_{}_{}_{}_{}-metrics".format(LEARNING_RATE, MODEL_NAME, "type_of_premise", component), "a")

    w.write("{},{},{},{},{},{},{}\n".format(str(acc), str(f1), str(precision), str(recall), str(f1_micro), str(precision_micro), str(recall_micro)))
    w.close()

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def labelAllExamples(filePatterns, type_of_prem):
    quadrant_types_to_label = {"fact": 0, "value": 1, "policy": 2}
    all_tweets = []
    all_labels = []
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            name_of_premise = ""
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
             # TODO: sacar todos los caracteres especiales
            tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "")
            type_of_quadrant = 0
            text_of_quadrant = ""
            is_argumentative = True
            filesize = 0
            for idx, word in enumerate(annotations):
                filesize += 1
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    if current_component.startswith("NonArgumentative"):
                       is_argumentative = False
                       break
                    if current_component.startswith(type_of_prem):
                        name_of_premise = ann[0]
                        text_of_quadrant += ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "")
                    elif current_component.split(" ")[0].startswith("QuadrantType"):
                        if name_of_premise == current_component.split(" ")[1]:
                            type_of_quadrant = quadrant_types_to_label[current_component.split(" ")[2].strip()]
                            break
            if filesize == 0 or not is_argumentative:
                continue
            tweet_text += " " + type_of_prem + ": " + text_of_quadrant
            processed_text = preprocessing.preprocess_text(tweet_text)
            all_tweets.append(preprocessed_text)
            all_labels.append(type_of_quadrant)
    ans = {"text": all_tweets, "label": all_labels}
    return Dataset.from_dict(ans)

def tokenize_preprocess(dataset, tokenizer):
    def tokenize_preprocess_per_example(example):
        return tokenizer(example["text"], truncation=True)

    return dataset.map(tokenize_preprocess_per_example, batched=True)


def train(epochs, model, tokenizer, train_partition_patterns, dev_partition_patterns, test_partition_patterns, premise):

    training_set = tokenize_preprocess(labelAllExamples(train_partition_patterns, premise), tokenizer)
    dev_set = tokenize_preprocess(labelAllExamples(dev_partition_patterns, premise), tokenizer)
    test_set = tokenize_preprocess(labelAllExamples(test_partition_patterns, premise), tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results_eval_{}_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME, "Type_of_premise", premise),
        evaluation_strategy="steps",
        eval_steps=20,
        save_total_limit=15,
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
        callbacks = [EarlyStoppingCallback(early_stopping_patience=30)]
    ) 

    trainer.train()
    print("--------------------------------------------------------------------------------------------------evaluation---------------------------------------------------------------------------------------------------------------------")
    print(trainer.evaluate())

    results = trainer.predict(test_set)
    filename = "./results_test_{}_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME, "Type_of_premise", premise)
    with open(filename, "w") as writer:
        print(results.metrics)
        writer.write("{},{},{},{}".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"]))


for premise in type_of_premises:
    component = premise
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)
    filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS+1)]
    train(0, model, tokenizer, filePatterns[:8], filePatterns[8:9], filePatterns[9:], premise)


