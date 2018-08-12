# Identifying GitHub Discussion @-Targets via Linguistic Analysis of Issue Posts

Jingxian Liao, Guowei Yang, David Kavaler, Vladimir Filkov and Prem Devanbu.

This repository includes all the code and data used in the manuscript.

## Data

Raw data are avaiable in [data folder](https://github.com/thalia-L/Identifying_GitHub_Discussion_Targets/tree/master/data) , including all the raw Issue thread posts on [sample_46_projs_issues](https://github.com/thalia-L/Identifying_GitHub_Discussion_Targets/tree/master/data/sample_46_projs_issues) and raw commit hisotry in [sample_46_projs_commit](https://github.com/thalia-L/Identifying_GitHub_Discussion_Targets/tree/master/data/sample_46_projs_issues). There are also closed class words list, Commonsets in both full dataset and committer-only data on the folder [wordlist](https://github.com/thalia-L/Identifying_GitHub_Discussion_Targets/tree/master/data/wordlist).

## Code

All the code is in [code folder](https://github.com/thalia-L/Identifying_GitHub_Discussion_Targets/tree/master/code). They need Keras and Tensorflow intalled and use Python3.

### Language model

[lm_deep_generic_write.py](https://github.com/thalia-L/Identifying_GitHub_Discussion_Targets/blob/master/code/lm_deep_generic_write.py) is written to build the language model and analyze important features in LM.

### sDAE-like model

sdae.py is used for sDAE-like model and the generation of specific embedding layer for SpokenP.
