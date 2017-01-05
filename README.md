# word2vec-theano
## DESCRIPTION : 
An implementation of the word2vec model in python using Theano.<br> 
I referred to the following resources for developing concepts and implementation details:<br>
#### 1. 'word2vec Parameter Learning Explained' by Xin Rong. <br>
#### 2. 'word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method' by Yoav Goldberg and Omer Levy.
The model I use here is Skip-Gram ( Section 3.2 in Mikolov et al. ). Further, the technique used for optimizing computational efficiency is Negative Sampling, the other option being Hierarchical Softmax which tends to be more complicated.
<br>
## MOTIVATION :
I had to understand the implementation details of Google's word2vec algrithm as first introduced in "Distributed Representations of Words and Phrases and their Compositionality" by Mikolov et al. 2013.
<br>
## PERFORMANCE :
I have trained my model on the The Harry Potter Series and it can be run through the API.<br>
The model can be tested from the IPython Notebook <b><u>runWord2Vec.ipynb</u></b>
<br>
##### mostSimilar('harry')
'ron'
##### mostSimilar('dumbledore')
'snape'
##### mostSimilar('wand')
'head'
##### mostSimilar('must')
'cant'
##### mostSimilar('have')
'has'
