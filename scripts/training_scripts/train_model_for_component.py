import glob
from sklearn.metrics import f1_score
import sys
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
from tqdm import tqdm
from transformers import EvalPrediction
from sklearn import metrics
import argparse
from transformers import EarlyStoppingCallback
from pysentimiento import preprocessing
import random

parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument('components', type=str, nargs='+', help="Name of the component that wants to be identified")
parser.add_argument('--modelname', type=str, default="roberta-base", help="Name of the language model to be downloaded from huggingface")
parser.add_argument('--lr', type=float, default=2e-05, help="Learning rate for training the model. Default value is 2e-05")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation. Default size is 16")
parser.add_argument('--add_annotator_info', type=bool, default=False, help="For Pivot and Collective add information about premises and Property respectively that an annotator would have when annotating these components")

args = parser.parse_args()


LEARNING_RATE = args.lr
NUMBER_OF_PARTITIONS = 10
device = "cuda:0"
BATCH_SIZE = args.batch_size
EPOCHS = 12 * (BATCH_SIZE / 16)
MODEL_NAME = args.modelname
REP=0
components = args.components
component = components[0]
add_annotator_info = args.add_annotator_info

def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids

    true_labels = [[str(l) for l in label if l != -100] for label in labels]
    true_predictions = [
        [str(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]


    all_true_labels = [l for label in true_labels for l in label]
    all_true_preds = [p for preed in true_predictions for p in preed]

    f1 = metrics.f1_score(all_true_labels, all_true_preds, average="macro")

    f1_binary = metrics.f1_score(all_true_labels, all_true_preds, average="binary", pos_label='1')

    acc = metrics.accuracy_score(all_true_labels, all_true_preds)

    recall = metrics.recall_score(all_true_labels, all_true_preds, average="binary", pos_label='1')

    precision = metrics.precision_score(all_true_labels, all_true_preds, average="binary", pos_label='1')

    f1_micro = metrics.f1_score(all_true_labels, all_true_preds, average="micro")

    recall_micro = metrics.recall_score(all_true_labels, all_true_preds, average="micro")

    precision_micro = metrics.precision_score(all_true_labels, all_true_preds, average="micro")

    confusion_matrix = metrics.confusion_matrix(all_true_labels, all_true_preds)


    w = open("./results_{}_{}_{}_{}_{}-metrics".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, component, REP), "a")

    w.write("{},{},{},{},{},{},{},{}\n".format(str(acc), str(f1), str(precision), str(recall), str(f1_micro), str(precision_micro), str(recall_micro), str(f1_binary)))
    w.close()

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': confusion_matrix,
        'f1_binary': f1_binary
    }



def labelComponents(text, component_text):
    if len(text.strip()) == 0:
        return []
    if len(component_text) == 0:
        return [0] * len(text.strip().split())

    if component_text[0] != "" and component_text[0] in text:
        parts = text.split(component_text[0])
        rec1 = labelComponents(parts[0], component_text[1:])
        rec2 = []
        if len(parts) > 2:
            rec2 = labelComponents(component_text[0].join(parts[1:]), component_text)
        else:
            rec2 = labelComponents(parts[1], component_text[1:])
        return rec1 + [1] * len(component_text[0].strip().split()) + rec2
    return [0] * len(text.strip().split())

def delete_unwanted_chars(text):
    return text.replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")

def labelComponentsFromAllExamples(filePatterns, component, multidataset = False, add_annotator_info = False):
    all_tweets = []
    all_labels = []
    if multidataset:
        datasets = []
    for f in filePatterns:
        #for f in glob.glob(filePattern):
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
             # TODO: sacar todos los caracteres especiales
            tweet_text = delete_unwanted_chars(tweet.read())
            component_text = []
            if add_annotator_info:
                if component == "Collective":
                    property_text = []
                if component == "pivot":
                    justification_text = []
                    conclusion_text = []
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
                    if current_component.startswith(component):
                        component_text.append([delete_unwanted_chars(ann[2].lstrip())])
                    if add_annotator_info:
                        if component == "Collective" and current_component.startswith("Property"):
                            property_text.append(delete_unwanted_chars(ann[2].lstrip()))
                        if component == "pivot" and current_component.startswith("Premise1Conclusion"):
                            conclusion_text.append(delete_unwanted_chars(ann[2].lstrip()))
                        if component == "pivot" and current_component.startswith("Premise2Justification"):
                            justification_text.append(delete_unwanted_chars(ann[2].lstrip()))


            if add_annotator_info:
                if component == "Collective":
                    tweet_text += " Property: " + " ".join(property_text)
                if component == "pivot":
                    tweet_text += " Just: " + " ".join(justification_text) + " Conc: " + " ".join(conclusion_text)
            preprocessed_text = preprocessing.preprocess_tweet(tweet_text, lang='en', user_token="@user", url_token="link", hashtag_token="hashtag")
            component_text = [preprocessing.preprocess_tweet(comp, lang='en') for comp in component_text]
            normalized_text = normalize_text(preprocessed_text, component_text)
            labels = labelComponents(" ".join(normalized_text), component_text)
            if not is_argumentative or filesize == 0:
                continue

            elif multidataset:
                dicc = {"tokens": [normalized_text], "labels": [labels]}
                datasets.append([Dataset.from_dict(dicc), normalized_text])
            else:
                assert(len(normalized_text) > 0)
                all_tweets.append(normalized_text)
                all_labels.append(labels)

    if multidataset:
        return datasets

    ans = {"tokens": all_tweets, "labels": all_labels}
    return Dataset.from_dict(ans)

def tokenize_and_align_labels(dataset, tokenizer, is_multi = False, is_bertweet=False):
    def tokenize_and_align_labels_per_example(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(example[f"labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


    def tokenize_and_align_labels_per_example_bertweet(example):
        tkns = example["tokens"]
        labels = example["labels"]
        if len(tkns) == 0 and len(labels) == 0:
            return {"input_ids": [], "labels": [], "attention_mask": []}
        tokenized_input = tokenizer(tkns, truncation=True, is_split_into_words=True)
        label_ids = [-100]
        for word, label in zip(tkns, labels):
            tokens = tokenizer(word).input_ids
            label_ids.append(label)
            for i in range(len(tokens)-3):
                label_ids.append(-100)
        label_ids.append(-100)
        assert(len(tokenized_input.input_ids) == len(label_ids))
        assert(len(tokenized_input.input_ids) == len(tokenized_input.attention_mask))
        return {"input_ids": tokenized_input.input_ids, "labels": label_ids, "attention_mask": tokenized_input.attention_mask}


    function_to_apply = tokenize_and_align_labels_per_example
    if is_bertweet:
        function_to_apply = tokenize_and_align_labels_per_example_bertweet
        if is_multi:
            return [{"dataset": data[0].map(function_to_apply), "text": data[1]} for data in dataset]
        return dataset.map(function_to_apply)
    if is_multi:
        return [{"dataset": data[0].map(function_to_apply, batched=True), "text": data[1]} for data in dataset]
    return dataset.map(function_to_apply, batched=True)

def normalize_text(tweet_text, arg_components_text):
    parts_processed = []
    splitted_text = [tweet_text]
    for splitter in arg_components_text:
        assert (splitter in tweet_text)
        new_splitted_text = []
        for segment in splitted_text:
            if segment != "":
                new_split = segment.split(splitter)
                for idx, splitt in enumerate(new_split):
                    new_splitted_text.append(splitt)
        splitted_text = new_splitted_text

    reconstructed_text = []
    current_text = tweet_text
    for part in splitted_text:
        if (part != ''):
            spp = current_text.split(part)
            for word in spp[0].split():
                reconstructed_text.append(word)
            for word in part.split():
                reconstructed_text.append(word)
            current_text = part.join(spp[1:])
    return reconstructed_text
    


    for idx, part in enumerate(splitted_text):
        for word in part.strip().split():
            parts_processed.append(word)

    return parts_processed


def train(epochs, model, tokenizer, train_partition_patterns, dev_partition_patterns, test_partition_patterns, component, is_bertweet=False, add_annotator_info=False):


    training_set = tokenize_and_align_labels(labelComponentsFromAllExamples(train_partition_patterns, component, add_annotator_info=add_annotator_info), tokenizer, is_bertweet = is_bertweet)
    dev_set = tokenize_and_align_labels(labelComponentsFromAllExamples(dev_partition_patterns, component, add_annotator_info=add_annotator_info), tokenizer, is_bertweet = is_bertweet)
    test_set = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component, add_annotator_info=add_annotator_info), tokenizer, is_bertweet = is_bertweet)
    test_set_one_example = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component, multidataset = True, add_annotator_info=add_annotator_info), tokenizer, is_multi = True, is_bertweet = is_bertweet)
    
    training_args = TrainingArguments(
        output_dir="./results_eval_{}_{}".format(MODEL_NAME.replace("/", "-"), component),
        evaluation_strategy="steps",
        eval_steps=10,
        save_total_limit=8,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.05,
        report_to="none",
        metric_for_best_model='f1_binary',
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= compute_metrics_f1,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    ) 

    trainer.train()

    results = trainer.predict(test_set)
    filename = "./results_test_{}_{}_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, REP, component)
    with open(filename, "w") as writer:
        writer.write("{},{},{},{},{}\n".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"], results.metrics["test_f1_binary"]))
        writer.write("{}".format(str(results.metrics["test_confusion_matrix"])))

    examples_filename = "./examples_test_{}_{}_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), BATCH_SIZE, REP, component)
    with open(examples_filename, "w") as writer:
        for dtset in test_set_one_example:
            result = trainer.predict(dtset["dataset"])
            preds = result.predictions.argmax(-1)[0]
            assert (len(preds) == len(result.label_ids[0]))
            comparison = [(truth, pred) for truth, pred in zip(result.label_ids[0], preds) if truth != -100]
            writer.write("Tweet:\n")
            writer.write("{}\n".format(dtset["dataset"]["tokens"][0]))
            for word, pair in zip(dtset["dataset"]["tokens"][0], comparison):
                writer.write("{}\t\t\t{}\t{}\n".format(word, pair[0], pair[1]))
            writer.write("-------------------------------------------------------------------------------\n")



filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS + 1)]

allFiles = []
for pattern in filePatterns:
    for f in glob.glob(pattern):
        allFiles.append(f)

dataset_combinations = []
for i in range(3):
    allFilesCp = allFiles.copy()    
    random.Random(41 + i).shuffle(allFilesCp)
    dataset_combinations.append([allFilesCp[:770], allFilesCp[770:870], allFilesCp[870:]])

# dataset_combination is a list of lists with three combinations of possible partitions for the dataset, being the first one a list with 8 folders of tweets used for training and the second and third lists with one folder of tweets used for eval and test
#dataset_combinations = [[filePatterns[2:], filePatterns[0:1], filePatterns[1:2]], [filePatterns[1:9], filePatterns[9:], filePatterns[:1]]]
#dataset_combinations = [[filePatterns[:8], filePatterns[8:9], filePatterns[9:]], [filePatterns[2:], filePatterns[0:1], filePatterns[1:2]], [filePatterns[1:9], filePatterns[9:], filePatterns[:1]]]
for combination in dataset_combinations:
    REP = REP + 1
    for cmpnent in components:
        component = cmpnent
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(device)
        train(0, model, tokenizer, combination[0], combination[1], combination[2], cmpnent, is_bertweet = MODEL_NAME == "bertweet-base", add_annotator_info=add_annotator_info)


