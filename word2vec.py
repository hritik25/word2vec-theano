""" 
Here, as I am following the paper titled "word2vec Parameter Learning Explained" by Xin Ron,
it will be referred to as [1] from here onward in subsequent comments for the code.
"""
import theano
import theano.tensor as T
import numpy as np
import createDataset
import math
import random
import cPickle


# dimension of the embedding vector
embeddingSize = 64
vocabSize = 10000


# Now,  we need to create a dataset from all words to use in training the model


# CREATE DATASET
print "creating dataset..." 
# vocabSize denotes the number of most common words to be used in word embeddings
data, counts, ranks, ranksToWords = createDataset.createDataset(vocabSize)


# class for a table which will be used to draw out negative samples.
class tableForNegativeSamples:
    def __init__(self, counts):
        # from Mikolov et al.'s original word2vec implementation, where they use 
        # a unigram distribution raised to power 3/4 to construct negative samples
        power = 0.75
        norm = sum([math.pow(t[1], power) for t in counts]) # Normalizing constants
        
        # tableSize should be big enough so that the minimum probability, i.e. 
        # (unigram)^(3/4) for a word multiplied by tableSize comes out to be atleast 1.
        tableSize = 1e8
        table = np.zeros(tableSize, dtype=np.uint16)

        p = 0 # Cumulative probability
        i = 0
        for word, count in counts:
            p += float(math.pow(count, power))/norm
            # fill the word in the table in the between the 
            # markings drawn out by cumulative probabilities
            while i < tableSize and float(i) / tableSize < p:
                table[i] = ranks[word]
                i += 1
        self.table = table

    def sample(self, k):
        indices = np.random.randint(low=0, high=len(self.table), size=k)
        return [self.table[i] for i in indices]


table = tableForNegativeSamples(counts)

dataIndex = 0
def generateBatch(positiveSampleSize, skipWindow, kNegativeSamples):
    """
    PARAMETERS : 
    positiveSampleSize - length of the window which will be sliding on the
                    continuous stream of words to generate a batch
    skipWindow - the number of context words to be considered on either 
                either side of the taget word
    
    RETURNS :
    batch - list of length = positiveSampleSize*(1 + kNegativeSamples)
            consisting of tuples (target, context) and including negative
            samples
    labels - list of 0s, 1s. 1 for positive sample, 0 for negative sample.
    """    
    global dataIndex
    assert positiveSampleSize % (2*skipWindow) == 0
    batch = []
    labels = []
    span = 2 * skipWindow + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[dataIndex])
        dataIndex = (dataIndex + 1) % len(data)
    for i in range(positiveSampleSize/(2*skipWindow)):
        context = skipWindow  # target label at the center of the buffer
        contextsToAvoid = [skipWindow]
        for j in range(2*skipWindow):
            while context in contextsToAvoid:
                context = random.randint(0, span-1)
            contextsToAvoid.append(context)
            positiveSample = (buffer[skipWindow], buffer[context])
            batch.append(positiveSample)
            labels.append(1)
            # attach negative samples
            negativeSamples = table.sample(kNegativeSamples)
            for i in range(kNegativeSamples):
                negativeSample = (buffer[skipWindow], negativeSamples[i])
                batch.append(negativeSample)
                labels.append(0)
        buffer.append(data[dataIndex])
        dataIndex = (dataIndex + 1) % len(data)
    return batch, labels


####################
### CREATE MODEL ###
####################
positiveSampleSize = 8
kNegativeSamples = 5
skipWindow = 1
batch, labels = generateBatch(positiveSampleSize=positiveSampleSize, skipWindow=skipWindow, 
                              kNegativeSamples = kNegativeSamples)

## SOME CONSTANTS 
dataIndex = 0  # will be modified as a global variable by the generateBatch function
batchSize = positiveSampleSize*(1 + kNegativeSamples)

###############################
### THE CORE IMPLEMENTATION ###
###############################

# the W matrix of the inputVectors as used in [1]
targetEmbeddings = theano.shared(np.random.uniform(-1, 1, (vocabSize, embeddingSize)))
# the W' matrix of the outputVectors as used in [1]
contextEmbeddings = theano.shared(np.random.normal(scale = 1.0/np.sqrt(vocabSize), 
                                                   size = (embeddingSize, vocabSize)))

# A |batchSize x 2| dimensional matrix, having (traget, context) pairs for
# a batch (including) -ve samples. This is the input to the training function .
targetContext = T.imatrix()

# the |batchSize x 1| vector, trainig labels (also an input to the training
# function), whether the context word matches the target word or not
isContext = T.bvector()

batchMatchScores = []

for i in range(batchSize):
    matchScore = T.dot(targetEmbeddings[targetContext[i][0],:], contextEmbeddings[:,targetContext[i][1]])
    batchMatchScores.append(matchScore)

objective = isContext*T.log(T.nnet.sigmoid(batchMatchScores)) + (1 - isContext)*T.log(1-T.nnet.sigmoid(batchMatchScores))

loss = -T.mean(objective)

# TRAINING FUNCTION
from lasagne.updates import nesterov_momentum
updates = nesterov_momentum(loss, [targetEmbeddings, contextEmbeddings], learning_rate = 0.1, momentum = 0.9)
trainBatch = theano.function([targetContext, isContext], loss, updates = updates)

numberOfBatches = len(data)/(positiveSampleSize/(2*skipWindow))

######################
### BEGIN TRAINING ###
######################
NUMBER_OF_EPOCHS = 10
print 'training start...'
print 'Total Number of Batches = ', numberOfBatches
for epoch in range(NUMBER_OF_EPOCHS):
    for i in range(numberOfBatches):
        batch, labels = generateBatch(positiveSampleSize, skipWindow, kNegativeSamples)
        batch = np.asarray(batch, dtype = np.uint16)
        labels = np.asarray(labels, dtype = np.int8)
        trainBatch(batch, labels)
        if i%1000 == 0:    
            print 'Batch {0} complete.'.format(i)
print 'training complete...'

# Save the word embeddings
with open('wordEmbeddings.pkl', 'wb') as file:
    cPickle.dump(targetEmbeddings.get_value(), file, protocol = 2)
