import glob

for f in glob.glob("/home/damifur/Escritorio/brat-v1.3_Crunchy_Frog/data1/HateEval/partition_1/*.ann"):
    annotations = open(f)
    tweet = open(f.replace(".ann", ".txt"))
    for idx, word in enumerate(annotations):
        ann = word.replace("\n", "").split("\t")
        if ann[1].lstrip().startswith("Collective"):
            tweet_final = []
            label_final = []
            for parts in tweet.read().split(ann[2]):
                print(parts)
            print(ann[2])
#        print(ann[1])
        print("--------------------------------------------------------------------------------------------")
