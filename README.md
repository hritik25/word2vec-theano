# word2vec-theano
## DESCRIPTION : 
An implementation of the word2vec model in python using Theano.<br> 
I referred to the following resources for developing concepts:<br> 
<ul>
<li><i>'word2vec Parameter Learning Explained'</i> by Xin Rong.<br>
<li><i>'word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method'</i> by Yoav Goldberg and Omer Levy.</ul>
The model I use here is Skip-Gram ( Section 3.2 in Mikolov et al. ). Further, the technique used for optimizing computational efficiency is Negative Sampling, the other option being Hierarchical Softmax which tends to be more complicated. Thanks to Edward Newell's <a href="http://cgi.cs.mcgill.ca/~enewel3/posts/implementing-word2vec/">blogpost</a> for a highly intuitive explaination. Here's the summary:<br>

1. word2vec is based on two assumptions:

  1. **distributional hypothesis** : meanings can be learned from the likely neighboring words.

  2. **meanings can be encoded in vectors**, and words with similar meanings should get similar vectors for example.

2. The learning objective and the Negative Sampling approach of word2vec is based on _Noise-Contrastive Estimation_. The idea is that we can learn a distribution by learning to distinguish it from other distributions. A _signal_ is defined as a pair(query-word, context-word) and _noise_ is defined as a pair(query-word, random-word). The signals are drawn from the natural distribution in the corpus, p(w,c) and the noises are drawn from some other distribution, p'(w,c). Let C indicate the truth value that a particular query-context pair \<w,c> is drawn from the natural distribution p(w,c), i.e C=1 if Yes and C=0 if No. So, Noise Contrastive approach will learn to model the probability:
> p(C=1 | w,c)

4. This can be converted to a supervised learning probem by first defining the 'match-score' as:
> p(C=1 | w,c) = σ(vw.vc)

where vw and vc are the word embedding vectors of w and c respectively and σ is the sigmoid function, and then defining the training objective as the cross-entropy loss:
> J = -( ∑\<w,c>( C\*ln( σ(vw.vc) ) + (1-C)\*ln( 1-σ(vw.vc) ) ) ) 

## MOTIVATION :
As a requisite to one other project of mine(<a href='https://github.com/hritik25/Dynamic-CNN-for-Modelling-Sentences/'>Dynamic CNN for modelling sentences</a>), I was required to understand the implementation details of Google's word2vec algrithm as first introduced in "Distributed Representations of Words and Phrases and their Compositionality" by Mikolov et al. 2013.
<br>
## PERFORMANCE :
I have trained my model on the The Harry Potter Series and it can be run from the IPython Notebook <b>runWord2Vec.ipynb</b>
<br><br>
`mostSimilar('harry')`

'ron'

`mostSimilar('dumbledoe')`

'snape'

`mostSimilar('wand')`

'head'

`mostSimilar('must')`

'cant'

`mostSimilar('have')`

'has'
