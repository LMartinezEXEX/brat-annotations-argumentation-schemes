import glob
from sklearn.metrics import cohen_kappa_score


all_tweets = []
labels_justifications_dami = []
labels_justifications_laura = []
labels_justifications_jose = []
for f in glob.glob("/home/developer_dami/brat-annotations-argumentation-schemes/data/HateEval/agreement_test_dami_only_arg/*.ann"):
    annotations = open(f, 'r')
    tweet = open(f.replace(".ann", ".txt"), 'r')
    tweet_text = tweet.read().replace("\n", "").replace("\t", "")
    there_is_justification = False
    for idx, word in enumerate(annotations):
        ann = word.replace("\n", "").split("\t")
        #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
        if ann[1].lstrip().startswith("Premise2Justification"):
            there_is_justification = True
            label_final = []
            partes = tweet_text.split(ann[2].lstrip().replace("\n", "").replace("\t", ""))
            for idx, part in enumerate(partes):
                for word in part.strip().split(" "):
                    if word != "":
                        label_final.append("0")
                if idx != len(partes) - 1:
                    for wd in ann[2].lstrip().split(" "):
                        if wd != "":
                            label_final.append("1")
#            print(tweet_final)
#            print(label_final)
            labels_justifications_dami.append(label_final)
    if there_is_justification:
        all_tweets.append(tweet_text.split(" "))
#        print(tweet_splited)
#        print("--------------------------------------------------------------------------------------------")

for labels in labels_justifications_dami:
    print(len(labels))
print("---------------------------------------------------------")
dami = [l for labels in labels_justifications_dami for l in labels]


all_tweets = []
for f in glob.glob("/home/developer_dami/brat-annotations-argumentation-schemes/data/HateEval/agreement_test_laura_only_arg/*.ann"):
    annotations = open(f, 'r')
    tweet = open(f.replace(".ann", ".txt"), 'r')
    tweet_text = tweet.read().replace("\n", "").replace("\t", "")
    there_is_justification = False
    for idx, word in enumerate(annotations):
        ann = word.replace("\n", "").split("\t")
        #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
        if ann[1].lstrip().startswith("Premise2Justification"):
            there_is_justification = True
            label_final = []
            partes = tweet_text.split(ann[2].lstrip().replace("\n", "").replace("\t", ""))
            for idx, part in enumerate(partes):
                for word in part.strip().split(" "):
                    if word != "":
                        label_final.append("0")
                if idx != len(partes) - 1:
                    for wd in ann[2].lstrip().split(" "):
                        if wd != "":
                            label_final.append("1")
#            print(tweet_final)
#            print(label_final)
            labels_justifications_laura.append(label_final)
    if there_is_justification:
        all_tweets.append(tweet_text.split(" "))
#        print(tweet_splited)
#        print("--------------------------------------------------------------------------------------------")

for labels in labels_justifications_laura:
    print(len(labels))
print("---------------------------------------------------------")
laura = [l for labels in labels_justifications_laura for l in labels]

all_tweets = []
for f in glob.glob("/home/developer_dami/brat-annotations-argumentation-schemes/data/HateEval/agreement_test_jose_only_arg/*.ann"):
    annotations = open(f, 'r')
    tweet = open(f.replace(".ann", ".txt"), 'r')
    tweet_text = tweet.read().replace("\n", "").replace("\t", "")
    there_is_justification = False
    for idx, word in enumerate(annotations):
        ann = word.replace("\n", "").split("\t")
        #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
        if ann[1].lstrip().startswith("Premise2Justification"):
            there_is_justification = True
            label_final = []
            partes = tweet_text.split(ann[2].lstrip().replace("\n", "").replace("\t", ""))
            for idx, part in enumerate(partes):
                for word in part.strip().split(" "):
                    if word != "":
                        label_final.append("0")
                if idx != len(partes) - 1:
                    for wd in ann[2].lstrip().split(" "):
                        if wd != "":
                            label_final.append("1")
#            print(tweet_final)
#            print(label_final)
            labels_justifications_jose.append(label_final)
    if there_is_justification:
        all_tweets.append(tweet_text.split(" "))
#        print(tweet_splited)
#        print("--------------------------------------------------------------------------------------------")


for labels in labels_justifications_jose:
    print(len(labels))
print(len(labels_justifications_jose))
jose = [l for labels in labels_justifications_jose for l in labels]

print(len(dami))
print(len(laura))
print(len(jose))

print(cohen_kappa_score(dami, jose))
print(cohen_kappa_score(dami, laura))
print(cohen_kappa_score(jose, laura))
