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

parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")
parser.add_argument('components', type=str, nargs='+', help="Name of the component that wants to be identified")
parser.add_argument('--modelname', type=str, default="roberta-base", help="Name of the language model to be downloaded from huggingface")
parser.add_argument('--lr', type=float, default=2e-05, help="Learning rate for training the model. Default value is 2e-05")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation. Default size is 16")

args = parser.parse_args()


LEARNING_RATE = args.lr
NUMBER_OF_PARTITIONS = 10
device = "cuda"
EPOCHS = 20
BATCH_SIZE = args.batch_size
MODEL_NAME = args.modelname
components = args.components
component = components[0]

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
    print("F1: {}".format(f1))

    f1_binary = metrics.f1_score(all_true_labels, all_true_preds, average="binary", pos_label='1')

    acc = metrics.accuracy_score(all_true_labels, all_true_preds)
    print("ACC: {}".format(acc))

    recall = metrics.recall_score(all_true_labels, all_true_preds, average="macro")
    print("Recall: {}".format(recall))

    precision = metrics.precision_score(all_true_labels, all_true_preds, average="macro")
    print("Precision: {}".format(precision))

    f1_micro = metrics.f1_score(all_true_labels, all_true_preds, average="micro")
    print(f1_micro)

    recall_micro = metrics.recall_score(all_true_labels, all_true_preds, average="micro")
    print(recall_micro)

    precision_micro = metrics.precision_score(all_true_labels, all_true_preds, average="micro")
    print(precision_micro)

    confusion_matrix = metrics.confusion_matrix(all_true_labels, all_true_preds)


    w = open("./results_{}_{}_{}-metrics".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), component), "a")

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


def labelComponentsFromAllExamples(filePatterns, component, multidataset = False):
    all_tweets = []
    all_labels = []
    if multidataset:
        datasets = []
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
             # TODO: sacar todos los caracteres especiales
            tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")
            component_text = []
            if component == "Collective":
                property_text = []
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
                        component_text.append([ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")])
                    if component == "Collective" and current_component.startswith("Property"):
                        property_text.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", ""))


            if component == "Collective":
                tweet_text += " Property: " + " ".join(property_text)
            preprocessed_text = preprocessing.preprocess_tweet(tweet_text, lang='en') + " " + f
            component_text = [preprocessing.preprocess_tweet(comp, lang='en') for comp in component_text] 
            normalized_text = normalize_text(preprocessed_text, component_text)
            labels = labelComponents(" ".join(normalized_text), component_text)
            if not is_argumentative or filesize == 0:
                continue

            elif multidataset:
                dicc = {"tokens": [normalized_text], "labels": [labels]}
                datasets.append([Dataset.from_dict(dicc), normalized_text])
            else:
                all_tweets.append(normalized_text)
                all_labels.append(labels)

    if multidataset:
        return datasets

    ans = {"tokens": all_tweets, "labels": all_labels}
    return Dataset.from_dict(ans)

def tokenize_and_align_labels(dataset, tokenizer, is_multi = False):
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

    if is_multi:
        return [{"dataset": data[0].map(tokenize_and_align_labels_per_example, batched=True), "text": data[1]} for data in dataset]
    return dataset.map(tokenize_and_align_labels_per_example, batched=True)

def normalize_text(tweet_text, arg_components_text):
    parts_processed = []
    splitted_text = [tweet_text]
    for splitter in arg_components_text:
        if (not splitter in tweet_text):
            print("ERRORRRRRRR")
            print(arg_components_text)
            print(splitter)
            print(tweet_text)
        assert (splitter in tweet_text)
        for segment in splitted_text:
            new_splitted_text = []
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


def train(epochs, model, tokenizer, train_partition_patterns, dev_partition_patterns, test_partition_patterns, component):

    training_set = tokenize_and_align_labels(labelComponentsFromAllExamples(train_partition_patterns, component), tokenizer)
    dev_set = tokenize_and_align_labels(labelComponentsFromAllExamples(dev_partition_patterns, component), tokenizer)
    test_set = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component), tokenizer)
    test_set_one_example = tokenize_and_align_labels(labelComponentsFromAllExamples(test_partition_patterns, component, multidataset = True), tokenizer, is_multi = True)
    
    training_args = TrainingArguments(
        output_dir="./results_eval_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), component),
        evaluation_strategy="steps",
        eval_steps=20,
        save_total_limit=15,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        report_to="none",
        metric_for_best_model='f1_binary',
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
    filename = "./results_test_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), component)
    with open(filename, "w") as writer:
        writer.write("{},{},{},{},{}\n".format(results.metrics["test_accuracy"], results.metrics["test_f1"], results.metrics["test_precision"], results.metrics["test_recall"], results.metrics["test_f1_binary"]))
        writer.write("{}".format(str(results.metrics["test_confusion_matrix"])))

    examples_filename = "./examples_test_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME.replace("/", "-"), component)
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




for cmpnent in components:
    component = cmpnent
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS + 1)]
    train(0, model, tokenizer, filePatterns[:8], filePatterns[8:9], filePatterns[9:], cmpnent)


