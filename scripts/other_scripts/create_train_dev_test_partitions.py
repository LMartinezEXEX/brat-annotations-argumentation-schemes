from glob import glob
import random
import shutil

filePatterns = ["./data/HateEval/partition_{}/hate_tweet_*.ann".format(partition_num) for partition_num in range(1, 11)]

txts = []
anns = []

for filePattern in filePatterns:
    for f in glob(filePattern):
        f_tweet = f.replace(".ann", ".txt")
        txts.append(f_tweet)
        anns.append(f)

combined = list(zip(txts, anns))
random.seed(99)
random.shuffle(combined)

print(len(combined))
print(combined[:3])

i = 1
for tweet, ann in combined[:770]:
    shutil.copyfile(tweet, "train_dataset/hate_tweet_{}.txt".format(str(i)))
    shutil.copyfile(ann, "train_dataset/hate_tweet_{}.ann".format(str(i)))
    i += 1

for tweet, ann in combined[770:870]:
    shutil.copyfile(tweet, "dev_dataset/hate_tweet_{}.txt".format(str(i)))
    shutil.copyfile(ann, "dev_dataset/hate_tweet_{}.ann".format(str(i)))
    i += 1

for tweet, ann in combined[870:]:
    shutil.copyfile(tweet, "test_dataset/hate_tweet_{}.txt".format(str(i)))
    shutil.copyfile(ann, "test_dataset/hate_tweet_{}.ann".format(str(i)))
    i += 1

