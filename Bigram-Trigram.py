def Bigram(normalized: list) -> list:
    """"given list it returns all possible bigrams"""

    bigram = [] #creating a list to append all the bigrams

    for index in range(len(normalized) - 1): # looping over the length -1 of the list to get the all possible bigrams
        bigrams = (normalized[index], normalized[index + 1]) #getting bigrams
        bigram.append(bigrams)

    return bigram


def Trigram(normalized: list) -> list:
    """given list it returns all possible trigrams"""

    trigram = [] #creating a list to append all the trigrams

    for index in range(len(normalized) - 2): # looping over the length -2 of the list to get the all possible trigrams
        trigrams = (normalized[index], normalized[index + 1], normalized[index + 2])
        trigram.append(trigrams)

    return trigram


def Fourgram(normalized: list) -> list:
    """given list, it returns all possible trigrams"""

    fourgram = [] #creating a list to append all the fourgrams

    for index in range(len(normalized) - 3): # looping over the length - 3 of the list to get the all possible fourgrams
        fourgrams = (normalized[index], normalized[index + 1], normalized[index + 2], normalized[index + 3])
        fourgram.append(fourgrams)

    return fourgram


def BigramFreq(bigram: list) -> list:
    """given bigrams, returns the frequency of the bigrams"""

    import nltk
    bigram_freqs = nltk.FreqDist(bigram)
    return bigram_freqs.most_common()


def TrigramFreq(trigram: list) -> list:
    """given trigrams, returns the frequency of the trigrams"""

    import nltk
    trigram_freq = nltk.FreqDist(trigram)
    return trigram_freq.most_common()


def FourgramFreq(fourgram: list) -> list:
    """given fourgrams, returns the frequency of fourgrams"""

    import nltk
    fourgram_freq = nltk.FreqDist(fourgram)
    return fourgram_freq.most_common()

