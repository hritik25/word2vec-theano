# word2vec-theano
DESCRIPTION : 
An implementation of the word2vec model in python using Theano. This is a direct implementation of the paper 'word2vec Parameter Learning Explained' by Xin Rong. Please note that it is a vanilla implementation ( i.e., without any special theano optimizations ), and the model I use here is Skip-Gram ( Section 3.2 in Mikolov et al. ). Further, the technique used for optimizing computational efficiency is Negative Sampling, the other option being Hierarchical Softmax which tends to be more complicated.

MOTIVATION :
I had to understand the implementation details of Google's word2vec algrithm as first introduced in "Distributed Representations of Words and Phrases and their Compositionality" by Mikolov et al. 2013.

APPLICATIONS :
I have trained my model on the The Harry Potter Series and it can be run through the API.
