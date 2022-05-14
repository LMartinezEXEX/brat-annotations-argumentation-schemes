import glob
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import sys

components = sys.argv[1:]

all_tweets = []
labels_justifications_dami = []
labels_justifications_jose = []

def labelComponents(text, justifications):
    if len(text.strip()) == 0:
        return []
    if len(justifications) == 0:
        return [0] * len(text.strip().split())

    if justifications[0] in text:
        parts = text.split(justifications[0])
        rec1 = labelComponents(parts[0], justifications[1:])
        rec2 = labelComponents(parts[1], justifications[1:])
        return rec1 + [1] * len(justifications[0].strip().split()) + rec2
    return [0] * len(text.strip().split())

filePatterns = ["./data/HateEval/agreement_tests/dami/*.ann", "./data/HateEval/agreement_tests/jose/*.ann"]

def labelComponentsFromAllExamples(filePattern, component):
    labels_per_example = []
    for f in glob.glob(filePattern):
        print(f)
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
        labels_per_example.append(labelComponents(tweet_text, component_text))

    return labels_per_example

# TODO
#def labelsAgreementOverlap(annotator1, annotator2, percentage):
#    counting = False
#    for an1, an2 in zip(annotator1, annotator2):
#        if an1 == 1 or ann2 == 1:
#            if not counting:
#                counting = True



for component in components:

    print("Data for component " + component)
    dami_examples = labelComponentsFromAllExamples(filePatterns[0], component)
    jose_examples = labelComponentsFromAllExamples(filePatterns[1], component)

    for dami, jose in zip(dami_examples, jose_examples):
        print("{} - {}".format(len(dami), len(jose)))

    dami_k = [l for label in dami_examples for l in label]
    jose_k = [l for label in jose_examples for l in label]

    print("Length of examples to compare (all the following numbers should be equal)")
    print(len(dami_k))
    print(len(jose_k))

    print("\n")
    print("Agreement between annotator 1 and 3")
    print(cohen_kappa_score(dami_k, jose_k))

    print("F1 score between annotators 1 and 3")
    print(f1_score(dami_k, jose_k))
