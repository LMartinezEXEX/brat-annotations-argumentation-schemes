import glob
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import sys
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


components = sys.argv[2:]
partition_num = sys.argv[1]

all_tweets = []
#labels_justifications_dami = []
#labels_justifications_laura = []
#labels_justifications_jose = []

def labelComponents(text, component_text):
    if len(text.strip()) == 0:
        return []
    if len(component_text) == 0:
        return [0] * len(text.strip().split())

    if component_text[0] != "" and component_text[0] in text:
        parts = text.split(component_text[0])
        rec1 = labelComponents(parts[0], component_text[1:])
        rec2 = labelComponents(parts[1], component_text[1:])
        return rec1 + [1] * len(component_text[0].strip().split()) + rec2
    return [0] * len(text.strip().split())

#filePatterns = ["./data/HateEval/agreement_test_dami_only_arg/*.ann", "./data/HateEval/agreement_test_laura_only_arg/*.ann", "./data/HateEval/agreement_test_jose_only_arg/*.ann"]

def labelComponentsFromAllExamples(filePattern, component):
    labels_per_example = []
    all_texts = []
    for f in glob.glob(filePattern):
        annotations = open(f, 'r')
        tweet = open(f.replace(".ann", ".txt"), 'r')
        # TODO: sacar todos los caracteres especiales
        tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "")
        component_text = []
        for idx, word in enumerate(annotations):
            ann = word.replace("\n", "").split("\t")
            #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
            if ann[1].lstrip().startswith(component):
                component_text.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", ""))

        
        normalized_text = normalize_text(tweet_text, component_text)
        labels = labelComponents(" ".join(normalized_text), component_text)
    
    
        all_texts.append(normalized_text)
        labels_per_example.append(labels)

    return all_texts, labels_per_example


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
# TODO
#def labelsAgreementOverlap(annotator1, annotator2, percentage):
#    counting = False
#    for an1, an2 in zip(annotator1, annotator2):
#        if an1 == 1 or ann2 == 1:
#            if not counting:
#                counting = True

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
filePattern = "./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num)
for component in components:

    text, labels = labelComponentsFromAllExamples(filePattern, component)
    print(labels)
    for t, l in zip(text, labels):
        print(tokenizer(t)['input_ids'])
        print(l)
