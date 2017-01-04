"""
The Harry Potter books were downloaded as text files from 
'https://github.com/abishekk92/potter/tree/master/dataset'
"""

from unidecode import unidecode # to stay out of all the decoding hassle !

import string

def produceWords():
    dataComplete = []
    for i in range(1,8):
        with open('dataset/book_' + str(i) + '.txt', 'r') as myfile:
            data=myfile.read()
            data = data.decode('utf-8')
            data = unidecode(data)
            data = data.translate(None, string.punctuation)
            dataComplete = dataComplete + data.lower().split()
    
    return dataComplete