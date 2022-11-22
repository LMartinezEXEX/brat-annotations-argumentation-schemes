# Argumentation Schemes for Automatic Counter-Narratives Generation Against Hate Speech #

## Overview ##
This repo contains a dataset of Hate Tweets against immigrants annotated with Argumentation Proto-Schemes and counter-narratives for those tweets classified in three different categories. Tweets were annotated using a [brat server][brat]. The brat server code is included on this repo so you can run it locally and easily inspect the dataset using your browser. Running a brat server locally can be also usefull to try or suggest changes on our annotations, adding new entities or changing the ones proposed by us. Please feel free to create branches from our repo and submit pull requests explaining why do you think the proposed changes can be of use.

[brat]:         http://brat.nlplab.org
[brat_repo]:    https://github.com/nlplab/brat/

## Where is the data? ##

The annotated hate tweets can be found in two different formats. Dataset in brat style format can be found in the "datasets" folder. Dataset in Conll format can be found in the "datasets_CONLL" folder. They are both divided in three partitions: train, dev and test. Brat style annotations contains two files per each tweet: a ".txt" file with the text of the tweet and a ".ann" file with the argumentative information and the counter-narratives associated to that tweet. CONLL style annotations also contains two files per tweet: a ".conll" file with the text of the tweet and the argumentative information in CONLL standard format and a ".cn" file with one counter-narrative per line per each type respecting the following order: the first line holds a counter-narrative of type A, the second line a counte-narrative of type B, the third a counter-narrative of type C and the fourth line a counter-narrative of type D. If the tweet doesn't have a counter-narrative of a specific type, the corresponding line is left blank.

If a brat server is run locally, data will be read from data/HateEval folder. Modifications or additions using the brat annotation tool will be reflected on that folder.

All tweets were extracted from [SemEval 2019 task 5] (a.k.a. HatEval) filtering only those that were non-aggressive, non-targeted against individuals and targeted against immigrants (HatEval also contains hateful tweets targeted agains women). For more details on how the dataset was constructed please check (and cite!) [this paper][https://arxiv.org/abs/2208.01099]

[SemEval 2019 task 5]:  https://aclanthology.org/S19-2007/

## Tools for using the data ##

This repo contains scripts that can be used to train models to recognice different types of argumentative information. They can also be used to reproduce the experiments described in the previously mentioned paper. Scripts works also for new types of annotations, so if you are interested on labelling a new or different kind or argumentative component you can use them. The repo also contains scripts for calculating statistics and agreement between annotators


## Ready to run ##
Brat server can be run locally after cloning this repo. To run it follow this steps:

1 - Rename the config_template.py file into config.py
    ``cp config_template.py config.py``
    
2 - Uptade the environment variables that you need. The only one needed to run brat is adding at least one user:password inside the USER_PASSWORD variable

3 - Run the server on localhost:
    ``python standalone.py``

This will automatically launch a server running on localhost on port 8080

The code for running this server was forked from [brat repo][brat_repo]. We want to thanks the brat developer team for creating this awsome annotation tool that we love.

## Citing ##

Damian A. Furman, Pablo Torres, Jose A. Rodriguez, Lautaro Martinez, Laura Alonso Alemany, Diego Letzen, Maria Vanina Martinez, 2022, Parsimonious Argument Annotations for Hate Speech Counter-narratives, https://arxiv.org/abs/2208.01099

## Contributing ##

Our dataset is a pilot test aimed to explore the role and impact of complex argumentation components (in this case, argumentation proto-schemes) on a fastly growing task like Automatic Counter-Narratives Generation. Since our work is continuously developing we encourage everyone to contribute with annotation of new examples, ideas for adding new components or new possible uses for argumentation schemes or proto-schemes in the context of Counter-Argumentation or Automatic Counter-narrative generation

## Contact ##

* Damián Furman       &lt;damian.a.furman@gmail.com&gt;
* Laura Alonso Alemany     &lt;lauraalonsoalemany@gmail.com&gt;
