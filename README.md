# tacl2015

Matlab code to train lexical and compositional RNN models.

To get started just run the scripts in the scripts directory to train models on word, bigram, or PPDB phrases.
Also word_code and single_W_code have testing scripts inside as well.

Options for training are specified in the config1.m files in word_code, bigram_code, and phrase_code.
These options include:

* hiddenSize : size of input word embeddings
* etat : reg parameter for composition matrix
* etaw : reg parameter for word embeddings
* margin : margin in objective function
* epochs : number of epochs to train
* save : whether to save models
* quiet : whether to suppress some output
* constraints : whether to train word model with lexical constraints
* wordfile : input embedding file
* output : output file name
* init : inital composition parameters

If you use our datasets and/or code for your work please cite:

@article{wieting2015ppdb,
title={From Paraphrase Database to Compositional Paraphrase Model and Back},
author={John Wieting and Mohit Bansal and Kevin Gimpel and Karen Livescu and Dan Roth},
journal={Transactions of the ACL (TACL)},
year={2015}}