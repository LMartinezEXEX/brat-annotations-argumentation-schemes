import glob

for f in glob.glob("/home/damifur/Escritorio/brat-v1.3_Crunchy_Frog/data1/HateEval/partition_1/*.ann"):
    annotations = open(f)
    tweet = open(f.replace(".ann", ".txt"))
    for idx, word in enumerate(annotations):
        ann = word.replace("\n", "").split("\t")
        if ann[1].lstrip().startswith("Collective"):
            tweet_final = []
            label_final = []
            partes = tweet.read().split(ann[2].lstrip())
            for idx, part in enumerate(partes):
                for word in part.strip().split(" "):
                    tweet_final.append(word)
                    label_final.append("0")
                if idx != len(partes) - 1:
                    for wd in ann[2].lstrip().split(" "):
                        tweet_final.append(wd)
                        label_final.append("1")
            print(len(tweet_final))
            print(len(label_final))
        print("--------------------------------------------------------------------------------------------")
