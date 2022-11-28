from glob import glob
from pysentimiento import preprocessing
import glob
import re

COMPONENTS = ["Property", "Collective", "Premise1Conclusion", "Premise2Justification", "pivot"]

def replaceSpace(string):
    pattern = " " + '{2,}'
    string = re.sub(pattern, " ", string)
    return string

def labelComponents(text, component_text, component):
    if len(text.strip()) == 0:
        return []
    if len(component_text) == 0:
        return ["O"] * len(text.strip().split())

    if component_text[0] != "" and component_text[0] in text:
        parts = text.split(component_text[0])
        rec1 = labelComponents(parts[0], component_text[1:], component)
        rec2 = []
        if len(parts) > 2:
            rec2 = labelComponents(component_text[0].join(parts[1:]), component_text, component)
        else:
            rec2 = labelComponents(parts[1], component_text[1:], component)
        return rec1 + ([component] * len(component_text[0].strip().split())) + rec2
    return ["O"] * len(text.strip().split())

def normalize_text(tweet_text, arg_components_text):
    splitted_text = [tweet_text]
    for splitter in arg_components_text:

        update_splitted_text = []
        if len(splitter.replace(" ", "")) > 0 and splitter.replace(" ", "") in tweet_text:
            tweet_text = tweet_text.replace(splitter.replace(" ", ""), splitter)
        for text_part in splitted_text:
            if len(splitter.replace(" ", "")) > 0 and splitter.replace(" ", "") in text_part:
                new_tweet_text = text_part.replace(splitter.replace(" ", ""), splitter)    
                update_splitted_text.append(new_tweet_text)
            else:
                update_splitted_text.append(text_part)
        splitted_text = update_splitted_text
                
        if splitter not in tweet_text and splitter.lower() in tweet_text:
            splitter = splitter.lower()
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


def labelComponentsFromAllExamples(filePatterns):
    for f in filePatterns:
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
            tweet_text = delete_unwanted_chars(tweet.read())
            all_component_text = []
            component_texts = {}
            is_argumentative = True
            filesize = 0
            name_of_premises = {}
            type_of_premises = {}
            for idx, word in enumerate(annotations):
                filesize += 1
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].lstrip()
                    if current_component.startswith("NonArgumentative"):
                        is_argumentative = False
                        break
                    if current_component.startswith("Premise"):
                        name_of_premises[ann[0]] = current_component.split(" ")[0]
                    if current_component.startswith("QuadrantType"):
                        info_splitted = current_component.split(" ")
                        type_of_premises[name_of_premises[info_splitted[1]]] = info_splitted[2]

                    new_component_list_aux = []
                    if current_component.startswith("Property") or current_component.startswith("Collective") or current_component.startswith("pivot") or current_component.startswith("Premise1Conclusion") or current_component.startswith("Premise2Justification"):
                        component_txt = preprocessing.preprocess_tweet(delete_unwanted_chars(ann[2].lstrip()), lang='en', user_token="@user", url_token="link", hashtag_token="hashtag")
                        new_component = component_txt
                        for cmpnt in COMPONENTS:
                            if current_component.startswith(cmpnt):
                                if cmpnt not in component_texts:
                                    component_texts[cmpnt] = []
                                component_texts[cmpnt].append(component_txt)
                        for component in all_component_text:
                            if component in new_component:
                                new_component = " ".join(normalize_text(new_component, [component]))
                            if new_component in component:
                                new_component_list_aux.append(normalize_text(component, [new_component]))
                            else:
                                new_component_list_aux.append(component)
                        all_component_text.append(new_component)


            preprocessed_text = preprocessing.preprocess_tweet(tweet_text, lang='en', user_token="@user", url_token="link", hashtag_token="hashtag")
            all_component_text = [preprocessing.preprocess_tweet(comp, lang='en', user_token="@user", url_token="link", hashtag_token="hashtag") for comp in all_component_text]
            normalized_text = normalize_text(preprocessed_text, all_component_text)
            assert(not (is_argumentative and ("Premise1Conclusion" not in type_of_premises or "Premise2Justification" not in type_of_premises)))

            component_labels = []
            argumentative = ["O" if is_argumentative else "NoArgumentative"] * len(normalized_text)
            component_labels.append(argumentative)
            for cmpnt in ["Premise2Justification", "Premise1Conclusion", "Collective", "Property", "pivot"]:
                if not is_argumentative:
                    labels = ["O"] * len(normalized_text)
                    type_of_justification = ["O"] * len(normalized_text)
                    type_of_conclusion = ["O"] * len(normalized_text)
                else:

                    if not cmpnt in component_texts:
                        labels = ["O"] * len(normalized_text)
                    else:
                        labels = labelComponents(" ".join(normalized_text), component_texts[cmpnt], cmpnt)
                    if cmpnt == "Premise2Justification":
                        type_of_justification = [type_of_premises[lbl] if lbl == "Premise2Justification" else "O" for lbl in labels]
                    elif cmpnt == "Premise1Conclusion":
                        type_of_conclusion = [type_of_premises[lbl] if lbl == "Premise1Conclusion" else "O" for lbl in labels]
                assert(len(normalized_text) == len(labels))
                component_labels.append(labels)


            component_labels.append(type_of_justification)
            component_labels.append(type_of_conclusion)

            conll = open(f.split("/")[-1].replace(".ann", ".conll"), "w")
            for idx, wrd in enumerate(normalized_text):
                line = [wrd]
                for i in range(len(component_labels)):
                    line.append(component_labels[i][idx])
                jointed = "\t".join(line)
                conll.write("{}\n".format(jointed))
            conll.close()




filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, 11)]

allFiles = []
for pattern in filePatterns:
    for f in glob.glob(pattern):
        allFiles.append(f)

labelComponentsFromAllExamples(allFiles)