# Argumentation Schemes for Automatic Counter-Narratives Generation Against Hate Speech #

## Overview ##
This repo contains a dataset of Hate Tweets against immigrants annotated with Argumentation Proto-Schemes and counter-narratives for those tweets classified in three different categories. Tweets were annotated using a [brat server][brat]. The brat server code is included on this repo so you can run it locally and easily inspect the dataset using your browser. Running a brat server locally can be also usefull to try or suggest changes on our annotations, adding new entities or changing the ones proposed by us. Please feel free to create branches from our repo and submit pull requests explaining why do you think the proposed changes can be of use.

[brat]:         http://brat.nlplab.org
[brat_repo]:    https://github.com/nlplab/brat/

## Ready to run ##
Brat server can be run locally after cloning this repo by typing:

    python standalone.py

This will automatically launch a server running on localhost on port 8080

The code for running this server was forked from [brat repo][brat_repo]. We want to thanks the brat developer team for creating this awsome annotation tool that we love.

## Where is the data? ##

The annotated hate tweets can be found on the data/HateEval folder. Dataset is organized in 11 partitions containing a total of 1084 tweets in English and one partition containing 196 tweets in Spanish. All tweets were extracted from [SemEval 2019 task 5] (a.k.a. HatEval) filtering only those that were non-aggressive, non-targeted against individuals and targeted against immigrants (HatEval also contains hateful tweets targeted agains women). For more details on how the dataset was constructed please check (and cite!) our soon to be published publication (the link will be uploaded very soon!).

[SemEval 2019 task 5]:  https://aclanthology.org/S19-2007/

## Tools for using the data ##

To Be completed soon!


## Citing ##

To Be completed soon!

## Contributing ##

Our dataset is a pilot test aimed to explore the role and impact of complex argumentation components (in this case, argumentation proto-schemes) on a fastly growing task like Automatic Counter-Narratives Generation. Since our work is continuously developing we encourage everyone to contribute with annotation of new examples, ideas for adding new components or new possible uses for argumentation schemes or proto-schemes in the context of Counter-Argumentation or Automatic Counter-narrative generation

## Contact ##

* Damián Furman       &lt;damian.a.furman@gmail.com&gt;
* Laura Alonso Alemany     &lt;lauraalonsoalemany@gmail.com&gt;
