from glob import glob
import random
import shutil
from pysentimiento import preprocessing
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import glob
import re

def replaceSpace(string):
    pattern = " " + '{2,}'
    string = re.sub(pattern, " ", string)
    return string

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

def normalize_text(tweet_text, arg_components_text):
    splitted_text = [tweet_text]
    for splitter in arg_components_text:

        if len(splitter.replace(" ", "")) > 0 and splitter.replace(" ", "") in tweet_text:
            tweet_text = tweet_text.replace(splitter.replace(" ", ""), splitter)
        if splitter not in tweet_text and splitter.lower() in tweet_text:
            splitter = splitter.lower()
        # print("-------------------------------------")
        # print(splitter)
        # print(tweet_text)
        assert (splitter in tweet_text)
        new_splitted_text = []
        for segment in splitted_text:
            if segment != "":
                new_split = segment.split(splitter)
                for idx, splitt in enumerate(new_split):
                    new_splitted_text.append(splitt)
                    if idx != len(new_split) -1:
                        new_splitted_text.append(splitter)
        splitted_text = new_splitted_text
        # print(splitted_text)

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

def delete_unwanted_chars(text):
    if re.match("[a-zA-Z]+#", text):
        text = text.replace("#", " #")
    text = " #".join(text.split("#"))
    return replaceSpace(text.lower().replace("\n", "").replace("\t", " ").replace(".", " ").replace(",", " ").replace("!", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", " ").replace("·", " ").replace(";", " "))


def labelComponentsFromAllExamples(filePatterns, multidataset = False, add_annotator_info = False):
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
            # if add_annotator_info:
            #     if component == "Collective":
            #         property_text = []
            #     if component == "pivot":
            #         justification_text = []
            #         conclusion_text = []
            is_argumentative = True
            filesize = 0
            for idx, word in enumerate(annotations):
                filesize += 1
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    new_component_list_aux = []
                    if current_component.startswith("Property") or current_component.startswith("Collective") or current_component.startswith("pivot") or current_component.startswith("Premise1Conclusion") or current_component.startswith("Premise2Justification"):
                        new_component = delete_unwanted_chars(ann[2].lstrip())
                        # if len(new_component) == 0:
                        #     print(ann)
                        for component in component_text:
                            if component in new_component:
                                new_component = " ".join(normalize_text(new_component, [component]))
                            if new_component in component:
                                new_component_list_aux.append(normalize_text(component, [new_component]))
                            else:
                                new_component_list_aux.append(component)
                        component_text.append(new_component)
                    # if add_annotator_info:
                    #     if component == "Collective" and current_component.startswith("Property"):
                    #         property_text.append(delete_unwanted_chars(ann[2].lstrip()))
                    #     if component == "pivot" and current_component.startswith("Premise1Conclusion"):
                    #         conclusion_text.append(delete_unwanted_chars(ann[2].lstrip()))
                    #     if component == "pivot" and current_component.startswith("Premise2Justification"):
                    #         justification_text.append(delete_unwanted_chars(ann[2].lstrip()))


            # if add_annotator_info:
            #     if component == "Collective":
            #         tweet_text += " Property: " + " ".join(property_text)
            #     if component == "pivot":
            #         tweet_text += " Just: " + " ".join(justification_text) + " Conc: " + " ".join(conclusion_text)
            preprocessed_text = preprocessing.preprocess_tweet(tweet_text, lang='en', user_token="@user", url_token="link", hashtag_token="hashtag")
            component_text = [preprocessing.preprocess_tweet(comp, lang='en', user_token="@user", url_token="link", hashtag_token="hashtag") for comp in component_text]
            # print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
            # print(component_text)
            # print(preprocessed_text)
            # print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
            # print(f)
            normalized_text = normalize_text(preprocessed_text, component_text)
            if preprocessed_text != " ".join(normalized_text):
                print("DIFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
                print(preprocessed_text)
                print(" ".join(normalized_text))

            # labels = labelComponents(" ".join(normalized_text), component_text)
    #         if not is_argumentative or filesize == 0:
    #             continue

    #         elif multidataset:
    #             dicc = {"tokens": [normalized_text], "labels": [labels]}
    #             datasets.append([Dataset.from_dict(dicc), normalized_text])
    #         else:
    #             assert(len(normalized_text) > 0)
    #             all_tweets.append(normalized_text)
    #             all_labels.append(labels)

    # if multidataset:
    #     return datasets

    # ans = {"tokens": all_tweets, "labels": all_labels}
    # return Dataset.from_dict(ans)




def train(epochs, model, tokenizer, train_partition_patterns, dev_partition_patterns, test_partition_patterns, is_bertweet=False, add_annotator_info=False):


    training_set = labelComponentsFromAllExamples(train_partition_patterns, add_annotator_info=add_annotator_info)
    dev_set = labelComponentsFromAllExamples(dev_partition_patterns, add_annotator_info=add_annotator_info)
    test_set = labelComponentsFromAllExamples(test_partition_patterns, add_annotator_info=add_annotator_info)
    test_set_one_example = labelComponentsFromAllExamples(test_partition_patterns, multidataset = True, add_annotator_info=add_annotator_info)
   






MODEL_NAME = "roberta-base"
filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, 11)]

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
    # REP = REP + 1
    # for cmpnent in components:
        # component = cmpnent
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
    # model.to(device)
    train(0, model, tokenizer, combination[0], combination[1], combination[2], is_bertweet = MODEL_NAME == "bertweet-base")








# filePatterns = ["./data/HateEval/partition_spanish/hate_tweet_*.ann"]

# txts = []
# anns = []

# for filePattern in filePatterns:
#     for f in glob(filePattern):
#         f_tweet = f.replace(".ann", ".txt")
#         txts.append(f_tweet)
#         anns.append(f)

# combined = list(zip(txts, anns))
# random.seed(99)
# random.shuffle(combined)

# print(len(combined))
# print(combined[:3])

# i = 1
# for tweet, ann in combined[:120]:
#     shutil.copyfile(tweet, "train_dataset_sp/hate_tweet_{}.txt".format(str(i)))
#     shutil.copyfile(ann, "train_dataset_sp/hate_tweet_{}.ann".format(str(i)))
#     i += 1

# for tweet, ann in combined[120:146]:
#     shutil.copyfile(tweet, "dev_dataset_sp/hate_tweet_{}.txt".format(str(i)))
#     shutil.copyfile(ann, "dev_dataset_sp/hate_tweet_{}.ann".format(str(i)))
#     i += 1

# for tweet, ann in combined[146:196]:
#     shutil.copyfile(tweet, "test_dataset_sp/hate_tweet_{}.txt".format(str(i)))
#     shutil.copyfile(ann, "test_dataset_sp/hate_tweet_{}.ann".format(str(i)))
#     i += 1

