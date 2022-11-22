
from nltk.translate.bleu_score import sentence_bleu
import glob
from sentence_transformers import SentenceTransformer, util
import gensim.downloader as api
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')

NUMBER_OF_PARTITIONS = 10
filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, NUMBER_OF_PARTITIONS + 1)]


def delete_unwanted_chars(text):
    return text.replace("\n", "").replace("\t", "").replace(".", "").replace(",", "").replace("!", "").replace("#", "").replace('“', '"').replace('”', '"').replace('…', '').replace("’", "").replace("–", " ").replace("‘", "").replace("—", "").replace("·", "")


def get_counter_narratives_map(filePatterns):
    positions = {"CounterNarrativeA": 0, "CounterNarrativeB": 1, "CounterNarrativeC": 2, "CounterNarrativeD": 3}

    counter_narratives = {}
    for filePattern in filePatterns:
        for f in glob.glob(filePattern):
            narrative_keys = ["", "", "", ""]
            cns = ["", "", "", ""]
            annotations = open(f, 'r')
            tweet = open(f.replace(".ann", ".txt"), 'r')
            tweet_text = delete_unwanted_chars(tweet.read())
            for idx, word in enumerate(annotations):
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].strip()
                    for k in positions:
                        v = positions[k]
                        if current_component.startswith(k):
                            narrative_keys[v] = ann[0]
            annotations = open(f, 'r')
            for idx, word in enumerate(annotations):
                ann = word.replace("\n", "").split("\t")
                if len(ann) > 1:
                    current_component = ann[1].split()
                    if current_component[0].strip() == "AnnotatorNotes":
                        for idx, key in enumerate(narrative_keys):
                            if current_component[1].strip() == key:
                                cns[idx] = ann[2].strip()
            dicckey = f.replace("./data/HateEval/partition_", '').replace("/hate_tweet_", "-").replace(".ann", "")
            counter_narratives[dicckey] = cns
    return counter_narratives


def evaluate_with_BLEU(cn, ground_truth):
    print(sentence_bleu(ground_truth, cn))

def evaluate_with_sentence_embeddings(cn, ground_truth):
    embeddings_truth = model.encode(ground_truth)
    embeddings_cn = model.encode([cn])
    cosine_scores = util.cos_sim(embeddings_cn, embeddings_truth)
    print(cosine_scores)

def evaluate_with_word_embeddings(cn, ground_truth):
    glove_vector = api.load('glove-twitter-200')
    cn_words = [word for word in cn.split() if word in glove_vector]
    word_embeddings_cn = np.mean(glove_vector[cn_words], axis=0)

    word_embeddings_truth = []
    for truth_cn in ground_truth:
        words = [word for word in truth_cn.split() if word in glove_vector]
        word_embeddings_truth.append(np.mean(glove_vector[words], axis=0))
    cosine_scores = util.cos_sim(word_embeddings_cn, np.array(word_embeddings_truth))
    print(cosine_scores)

def evaluate_cn(tweet_number, cn):
	cn_map = get_counter_narratives_map(filePatterns)

	cns = cn_map[tweet_number]
	not_empty_cns = []
	for cnn in cns:
		if cnn != "":
			not_empty_cns.append(cnn)

	evaluate_with_sentence_embeddings(cn, not_empty_cns)
	evaluate_with_word_embeddings(cn, not_empty_cns)


evaluate_cn("1-8", "If the streets of the UK are like that is not because of immigrants but because of poverty. Don't blame those who don't have nothing")