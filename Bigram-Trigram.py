sent = "I am human and I have a human heart which is not the same as having a mind"
sent = sent.split()

def Bigram(normalized):
    """"given list it returns all possible bigrams"""

    bigram = [] #creating a list to append all the bigrams
    for index in range(len(normalized) - 1): # looping over the length -1 of the list to get the all possible bigrams
        bigrams = (normalized[index], normalized[index + 1]) #getting bigrams
        bigram.append(bigrams)
    return bigram



def Trigram(normalized):
    """given list it returns all possible trigrams"""

    trigram = []
    for index in range(len(normalized) - 2):
        trigrams = (normalized[index], normalized[index + 1], normalized[index + 2])
        trigram.append(trigrams)
    return trigram



def BigramFreq(bigram):
    """given bigrams, returns the frequency of the bigrams"""

    import nltk
    bigram_freqs = nltk.FreqDist(bigram)
    return bigram_freqs.most_common()



def TrigramFreq(trigram):
    """given trigrams, returns the frequency of the trigrams"""

    import nltk
    trigram_freq = nltk.FreqDist(trigram)
    return trigram_freq.most_common()


