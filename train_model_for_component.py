import glob
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import sys
from transformers import RobertaConfig, RobertaForTokenClassification, AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
import torch
from tqdm import tqdm
from transformers import EvalPrediction
from sklearn import metrics

LEARNING_RATE = 2e-05
NUMBER_OF_PARTITIONS = 10
device = "cuda"
EPOCHS = 6
MODEL_NAME = "roberta-base"
components = sys.argv[2:]
to_delete = sys.argv[1]
component = components[0]

def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    print(labels.shape)
    print(preds.shape)

    true_labels = [[str(l) for l in label if l != -100] for label in labels]
    true_predictions = [
        [str(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]


    all_true_labels = [l for label in true_labels for l in label]
    all_true_preds = [p for preed in true_predictions for p in preed]

    f1 = metrics.f1_score(all_true_labels, all_true_preds, average="macro")
    print(f1)

    acc = metrics.accuracy_score(all_true_labels, all_true_preds)
    print(acc)

    recall = metrics.recall_score(all_true_labels, all_true_preds, average="macro")
    print(recall)

    precision = metrics.precision_score(all_true_labels, all_true_preds, average="macro")
    print(precision)

    w = open("./results_{}_{}_{}-metrics".format(LEARNING_RATE, MODEL_NAME, component), "w")

    w.write("Accuracy\n")
    w.write(str(acc))
    w.write("\n")
    w.write("F1\n")
    w.write(str(f1))
    w.write("\n")
    w.write("Precision\n")
    w.write(str(precision))
    w.write("\n")
    w.write("recall\n")
    w.write(str(recall))
    w.write("\n")

    w.close()

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#class TweetComponentDataset(Dataset):
#    def __init__(self, tweets, labels):
#        self.texts = tweets
#        self.labels = labels
#
#    def __len__(self):
#        return len(self.texts)
#
#    def __getitem__(self, index):
#        text = str(self.texts[index])
#        text = " ".join(text.split()) #Esto puede ser que esté repetido pero no está de más hacerlo acá
#
#        return {
#            'id': index,
#            'text': text,
#            'labels': self.labels[index],
#        }






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

#filePatterns = ["./data/HateEval/agreement_test_dami_only_arg/*.ann", "./data/HateEval/agreement_test_laura_only_arg/*.ann", "./data/HateEval/agreement_test_jose_only_arg/*.ann"]

def labelComponentsFromAllExamples(filePatterns, component):
    all_tweets = []
    all_labels = []
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
             # TODO: sacar todos los caracteres especiales
            tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "")
            component_text = []
            is_argumentative = True
            for idx, word in enumerate(annotations):
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    if current_component.startswith("NonArgumentative"):
                        is_argumentative = False
                        break
                    if ann[1].lstrip().startswith(component):
                        component_text.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", ""))
        
            if not is_argumentative:
                continue
            else:
                normalized_text = normalize_text(tweet_text, component_text)
                labels = labelComponents(" ".join(normalized_text), component_text)
    
                all_tweets.append(normalized_text)
                all_labels.append(labels)
    ans = {"tokens": all_tweets, "labels": all_labels}
    return Dataset.from_dict(ans)

def tokenize_and_align_labels(dataset, tokenizer):
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

    return dataset.map(tokenize_and_align_labels_per_example, batched=True)

def normalize_text(tweet_text, arg_components_text):
    parts_processed = []
    splitted_text = [tweet_text]
    for splitter in arg_components_text:
        new_splitted_text = []
        segment = splitted_text[-1]
        new_split = segment.split(splitter)
        for idx, splitt in enumerate(new_split):
            new_splitted_text.append(splitt)
            if idx != len(new_split)-1:
                new_splitted_text.append(splitter)
        splitted_text = new_splitted_text

 #           print(splitted_text)

    for idx, part in enumerate(splitted_text):
        for word in part.strip().split():
            parts_processed.append(word)

    return parts_processed


def train(epochs, model, tokenizer, train_partition_patterns, dev_partition_patterns, component):

#        tr_loss = 0
#        n_correct = 0
#        nb_tr_steps = 0
#        nb_tr_examples = 0
    training_set = tokenize_and_align_labels(labelComponentsFromAllExamples(train_partition_patterns, component), tokenizer)
    test_set = tokenize_and_align_labels(labelComponentsFromAllExamples(dev_partition_patterns, component), tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results_{}_{}_{}".format(LEARNING_RATE, MODEL_NAME, component),
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_set,
        eval_dataset=test_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics= compute_metrics_f1
    ) 

    trainer.train()
    print("--------------------------------------------------------------------------------------------------evaluation---------------------------------------------------------------------------------------------------------------------")
    print(trainer.evaluate())
    model.save_pretrained("TEST")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)
filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS)]

for cmpnent in components:
    component = cmpnent
    train(0, model, tokenizer, filePatterns[:8], filePatterns[8:], cmpnent)


#test_text = "Immigrants bring crime to our society. They are bad. They are a bad influence. They represent a cost to the taxpayer. Crimes are commited by them every day. Something must be made. I have an ice cream cone on top of my head".split()
#test_text = "Stop being spineless and tell the Eu its our country and stop migrants dictating to us via eu".split()

print("HERE")
#test_set = TweetComponentDataset([test_text], [[1] * len(test_text)], tokenizer, 280)

#test_params = {'batch_size': 1,
#            'shuffle': True,
#            'num_workers': 1
#}

#test_loader = DataLoader(test_set, **test_params) 
#with torch.no_grad():
#    model.eval()
#    for _,data in tqdm(enumerate(test_loader, 0)):
#        ids = data['ids'].to(device, dtype = torch.long)
#        mask = data['mask'].to(device, dtype = torch.long)
#        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#
#        outputs = model(input_ids=ids,attention_mask=mask, token_type_ids=token_type_ids)
#
#        predicted_token_class_ids = outputs.logits.argmax(-1)
#
#        print(predicted_token_class_ids)
#        print(predicted_token_class_ids.shape)
