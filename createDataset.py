import dataUtils
import collections

words = dataUtils.produceWords()
# At this point, we have a list 'words' of all words present in our text.

def createDataset(vocabSize):
    """
    RETURN :  
    
    data- 
        A list of tags for each word, denoting its rank by count in the corpus of text availbale.
        By rank I mean a word's place when they have been sorted according to their count of occurences in the text.
        So for example, if the word 'I' has rank 20 and the word 'am' has rank 60, then a portion of the text 'I am'
        will be returned as [ 20, 60 ]
    """
    # I will use 0 for rank of words not in the top <vocabSize> words by their counts, and call them 'rare words'
    
    # here I use the symbol 'UNK' for the rare words
    counts = [['UNK', -1]]
    counts.extend(collections.Counter(words).most_common(vocabSize - 1))
    ranks = dict()
    for word, count in counts:
        ranks[word] = len(ranks)
    data = list()
    # for keeping count of rare words
    rareCount = 0
    for word in words:
        if word in ranks:
            index = ranks[word]
        else:
            index = 0  # ranks['UNK']
            rareCount += 1
        data.append(index)
    counts[0][1] = rareCount
    ranksToWords = dict(zip(ranks.values(), ranks.keys()))
    return data, counts, ranks, ranksToWords
