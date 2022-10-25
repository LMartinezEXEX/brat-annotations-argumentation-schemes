
import csv
import glob
import os
import sys
import torch
import numpy as np
from torch import IntTensor
import gensim.downloader as api
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
w_embeddings = api.load('glove-twitter-200')

def check_input_file(filepath):
    if not os.path.isfile(filepath):
        sys.exit('ERROR: File {} doesn\'t exist'.format(filepath))
    if not filepath.endswith('.tsv'):
        sys.exit('ERROR: File {} should be .tsv extended'.format(filepath))
    if os.stat(filepath).st_size == 0:
        sys.exit('ERROR: File {} is empty'.format(filepath))

def get_language(argv):
    if len(argv) < 3:
        return 'english'
    return 'spanish' if argv[2] == '--sp' or argv[2] == '-spanish' else 'english'

def get_tweet_id(filepath):
    return  int(filepath.replace('./datasets_CoNLL/', '').replace('english/', '')       \
                        .replace('spanish/', '').replace('dev_dataset', '')             \
                        .replace('train_dataset', '').replace('test_dataset', '')       \
                        .replace('_sp', '').replace('/', '').replace('hate_tweet_', '') \
                        .replace('.cn', ''))

def get_counter_narratives_map(eng_partition = True):
    files_path = "./datasets_CoNLL/{}/**/*.cn".format('english' if eng_partition else 'spanish')

    counter_narratives_map = {}
    for file in glob.glob(files_path, recursive=True):
        tweet_id = get_tweet_id(file)

        with open(file, 'r') as cn_file:
            cns = [line.rstrip() for line in cn_file.readlines()]
        
        if tweet_id not in counter_narratives_map:
            counter_narratives_map[tweet_id] = cns
        else:
            sys.exit('ERROR: Counter-narratives with same id found. Id: {}'.format(tweet_id))
    
    return counter_narratives_map

def get_cn_from_input_file(filepath):
    counter_narratives_map = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for index, row in enumerate(reader):
            if len(row) == 0:
                continue
            if len(row) != 2:
                sys.exit('ERROR: Line {} in file must have only 2 elements: tweet_id <tab> counter-narrative [Found {} elements]'.format(index+1, len(row)))
            
            try:
                tweet_id = int(row[0])
            except ValueError:
                sys.exit('ERROR: Line {} tweet id is not a valid number [found: {}]'.format(index+1, row[0]))
            cn = row[1]
            if len(cn) == 0:
                print('WARNING: Line {} counter-narrative for tweet with id: {} is empty'.format(index+1, tweet_id))
            
            if tweet_id not in counter_narratives_map:
                counter_narratives_map[tweet_id] = cn
            else:
                print('WARNING: Line {} repeated tweet id [{}] found. Will be ignored'.format(index+1, tweet_id))
    
    return counter_narratives_map

def evaluate_with_BLEU(cn, ground_truth):
    splitted_cn = [word for word in cn.split()]
    splitted = [[word for word in cn_splitted] for cn_splitted in (cn.split() for cn in ground_truth)]
    return sentence_bleu(splitted, splitted_cn)

def evaluate_with_sentence_embeddings(cn, ground_truth):
    embeddings_truth = model.encode(ground_truth)
    embeddings_cn = model.encode([cn])
    cosine_scores = util.cos_sim(embeddings_cn, embeddings_truth)
    return IntTensor.item(torch.max(cosine_scores))

def evaluate_with_word_embeddings(tweet_id, cn, ground_truth):
    cn_words = [word for word in cn.split() if word in w_embeddings]
    if not cn_words:
        print('WARNING: No word embeddings found for words in tweet with id {}'.format(tweet_id))
        return 0.0
    word_embeddings_cn = np.mean(w_embeddings[cn_words], axis=0)

    word_embeddings_truth = []
    for truth_cn in ground_truth:
        words = [word for word in truth_cn.split() if word in w_embeddings]
        word_embeddings_truth.append(np.mean(w_embeddings[words], axis=0))
    cosine_scores = util.cos_sim(word_embeddings_cn, np.array(word_embeddings_truth))
    return IntTensor.item(torch.max(cosine_scores))

def save_scores(tweet_id, sent_score, bleu_score, word_score):
    with open('./scores.tsv', 'a') as output:
        output.write(str(tweet_id) + '\t' + str(sent_score) + '\t' + str(bleu_score) + '\t' + str(word_score) + '\n')

def main(argv):
    if argv is None:
        argv = sys.argv
    
    if len(argv) <= 1:
        sys.exit('Usage: python evaluation_script.py cn_file.tsv [-spanish, --sp]')
    input_file = argv[1]
    check_input_file(input_file)
    
    language = get_language(argv)
    print('--- Using {} partition\n'.format(language.upper()))
    use_english_dataset = language == 'english'
    cn_map = get_counter_narratives_map(use_english_dataset)
    input_cn_map = get_cn_from_input_file(input_file)

    for tweet_id in input_cn_map:
        if tweet_id not in cn_map:
            print('WARNING: Tweet with id {} doesn\'t exist in the {} partition'.format(tweet_id, language.upper()))
            continue

        non_empty_cns = [cn for cn in cn_map[tweet_id] if cn != '']
        if not non_empty_cns:
            print('WARNING: Tweet with id {} does not have golden counter-narratives to compare with'.format(tweet_id))
            save_scores(tweet_id, 0.0, 0.0, 0.0)
            continue
        
        print('Evaluating tweet {} ~~~'.format(tweet_id))
        sent_score = evaluate_with_sentence_embeddings(input_cn_map[tweet_id], non_empty_cns)
        bleu_score = evaluate_with_BLEU(input_cn_map[tweet_id], non_empty_cns)
        word_score = evaluate_with_word_embeddings(tweet_id, input_cn_map[tweet_id], non_empty_cns)
        save_scores(tweet_id, sent_score, bleu_score, word_score)
        

    

if __name__ == '__main__':
    sys.exit(main(sys.argv))