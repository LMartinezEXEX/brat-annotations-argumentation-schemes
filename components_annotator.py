import glob
from sklearn.metrics import cohen_kappa_score


all_tweets = []
labels_justifications_dami = []
labels_justifications_laura = []
labels_justifications_jose = []

def labelComponents(text, justifications):
    if len(text) == 0:
        return []
    if len(justifications) == 0:
        return [0] * len(text.lstrip().rstrip().split(" "))

    if justifications[0] in text:
        parts = text.split(justifications[0])
        rec1 = labelComponents(parts[0], justifications[1:])
        rec2 = labelComponents(parts[1], justifications[1:])
        print(parts)
        return rec1 + [1] * len(justifications[0].split(" ")) + rec2
    return []

for f in glob.glob("/home/developer_dami/brat-annotations-argumentation-schemes/data/HateEval/agreement_test_dami_only_arg/*.ann"):
    annotations = open(f, 'r')
    tweet = open(f.replace(".ann", ".txt"), 'r')
    tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "")
    justifications = []
    for idx, word in enumerate(annotations):
        ann = word.replace("\n", "").split("\t")
        #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
        if ann[1].lstrip().startswith("Collective"):
            justifications.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", ""))
    print(tweet_text)
    print(labelComponents(tweet_text, justifications))
    labels_justifications_dami.append(labelComponents(tweet_text, justifications))


for f in glob.glob("/home/developer_dami/brat-annotations-argumentation-schemes/data/HateEval/agreement_test_laura_only_arg/*.ann"):
    annotations = open(f, 'r')
    tweet = open(f.replace(".ann", ".txt"), 'r')
    tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "")
    justifications = []
    for idx, word in enumerate(annotations):
        ann = word.replace("\n", "").split("\t")
        #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
        if ann[1].lstrip().startswith("Collective"):
            justifications.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", ""))
    labels_justifications_laura.append(labelComponents(tweet_text, justifications))
 

for f in glob.glob("/home/developer_dami/brat-annotations-argumentation-schemes/data/HateEval/agreement_test_jose_only_arg/*.ann"):
    annotations = open(f, 'r')
    tweet = open(f.replace(".ann", ".txt"), 'r')
    tweet_text = tweet.read().replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "")
    justifications = []
    for idx, word in enumerate(annotations):
        ann = word.replace("\n", "").split("\t")
        #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
        if ann[1].lstrip().startswith("Collective"):
            justifications.append(ann[2].lstrip().replace("\n","").replace("\t", "").replace(".", "").replace(",", "").replace("!", ""))
 
    labels_justifications_jose.append(labelComponents(tweet_text, justifications))


dami = [l for labels in labels_justifications_dami for l in labels]
laura = [l for labels in labels_justifications_laura for l in labels]
jose = [l for labels in labels_justifications_jose for l in labels]

print(len(dami))
print(len(laura))
print(len(jose))

print(cohen_kappa_score(dami, jose))
print(cohen_kappa_score(dami, laura))
print(cohen_kappa_score(jose, laura))
