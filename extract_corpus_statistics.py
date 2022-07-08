import glob
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
import sys

NUMBER_OF_PARTITIONS = 10

components = sys.argv[1:]

all_tweets = []
labels_justifications_dami = []
labels_justifications_jose = []


filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS + 1)]

def delete_unwanted_chars(text):
        return text.replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")

def labelComponentsFromAllExamples(filePattern, component):
    non_argumentatives = 0
    have_component = 0
    component_words = 0
    all_words = 0
    for idxx, f in enumerate(glob.glob(filePattern)):
        print("{}: {}".format(idxx, f))
        annotations = open(f, 'r')
        tweet = open(f.replace(".ann", ".txt"), 'r')
        # TODO: sacar todos los caracteres especiales
        tweet_text = delete_unwanted_chars(tweet.read())
        all_words += len(tweet_text.split())
        component_text = []
        is_argumentative = True
        for idx, word in enumerate(annotations):
            ann = word.replace("\n", "").split("\t")
            #TODO: Si la justificacion esta dividida en mas de una parte esto no va a funcionar. Tampoco funciona si la anotacion corta una palabra a la mitad (por ejemplo, deja el punto final afuera)
            if len(ann) > 1:
               current_component = ann[1].lstrip()
               if current_component.startswith("NonArgumentative"):
                   is_argumentative = False
                   break
               if current_component.startswith(component):
                   print("insie")
                   print(delete_unwanted_chars(ann[2]))
                   component_text.append(delete_unwanted_chars(ann[2]))
                   print(component_text)
            else:
                print("NOT LONG ENOUGH")
                print(f)
                print(ann)

        if not is_argumentative:
            print("NOT ARGGGG")
            non_argumentatives += 1
        elif len(component_text) > 0:
            print("COMPONENT TEXT")
            print(component_text)
            have_component += 1
            component_words += len([word for component in component_text for  word in component.split()])

    return [non_argumentatives, have_component, component_words, all_words]

# TODO
#def labelsAgreementOverlap(annotator1, annotator2, percentage):
#    counting = False
#    for an1, an2 in zip(annotator1, annotator2):
#        if an1 == 1 or ann2 == 1:
#            if not counting:
#                counting = True



non_argumentative = 0
have = {}
words = {}
all_words = 0
for component in components:
    have[component] = 0
    words[component] = 0
    non_argumentative = 0
    all_words = 0
    print("Data for component " + component)
    for partition in filePatterns:
        results = labelComponentsFromAllExamples(partition, component)
        print(partition)
        print(results)
        non_argumentative += results[0]
        have[component] += results[1]
        words[component] += results[2]
        all_words += results[3]
    
print("Total words")
print(all_words)

print("Non argumentative")
print(non_argumentative)

print("Presence")
print(have)

print("words")
print(words)
