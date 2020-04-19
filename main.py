from textblob import TextBlob, Word
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
# nltk.download()
import urllib.request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import pandas as pd

# reading from the web
response = urllib.request.urlopen('http://localhost/github/btt/carReview.php')
html = response.read()
# print(html)

# cleaning html tagova
soup = BeautifulSoup(html, 'html5lib')
text = soup.get_text()
text2 = text.replace("    ", "")
# text2 = text2.split('\n')
# print(text2)

# brisanje sve sto nije letter
letters_only = re.sub("[^a-zA-Z]", " ", text2)
letters_only = letters_only.replace("   ", "\n")
lines = letters_only.splitlines()
# print(lines)


# brisanje viska whitespace
lines2 = []

for sentence in lines:
    lines2.append(sentence.strip())
# print(lines2)

# brisanje prazan string
linesFinal = list(filter(None, lines2))
# print(linesFinal)

# Lower case
lower_case = [x.lower() for x in linesFinal]
# print(lower_case)

"""OVO NIJE POTREBNO
# convert to tuple
tuple_all = tuple(lower_case)
# print(tuple_all)

# po 2 elementa u tuple
tuple_two_elements = tuple(tuple_all[x:x + 2]
                           for x in range(0, len(tuple_all), 2))
# print(tuple_two_elements)

# feedback bez pos i neg
feedbacks = [x[0] for x in tuple_two_elements]
# print(feedbacks)

# pos or neg
pos_neg = [x[1] for x in tuple_two_elements]
# print(pos_neg)
"""

# word tokenization
""" words = lower_case[0].split(' ')
print(words) """

# from nltk.tokenize import word_tokenize


wordToken = []
sentenceToken = []

for a in lower_case:
    sentenceToken.append(a)
    for i, word in enumerate(a.split()):
        wordToken.append(word)

# print(sentenceToken)
# print(wordToken)

# spelling coorection

count = 0
for x in wordToken:
    # print(x)
    w = Word(x)
    # print(w.spellcheck())
    if (w.spellcheck()[0][1] != 1):
        print("\n Incorrect word '" + w +
              "' --- Corrent word: '" + w.correct() + "' \n")

    """ for i in sentenceToken:
            if (w in sentenceToken[count]):
                sentenceToken[count] = TextBlob(sentenceToken[count]).correct()
            count = count + 1
        count = 0 """

correntWord = []
for x in wordToken:
    correntWord.append(TextBlob(x).correct())
    count = count + 1
# print(correntWord)

correntSentence = []
for i in sentenceToken:
    correntSentence.append(TextBlob(i).correct())
    count = count + 1
# print(correntSentence)


# word tokenizer 2
""" tokens = [t for t in wordToken]
print(tokens) """

sr = stopwords.words('english')
clean_tokens = correntWord[:]
for token in correntWord:
    if token in stopwords.words('english'):

        clean_tokens.remove(token)

# print(clean_tokens)

# frequency
freq = nltk.FreqDist(clean_tokens)
""" for key, val in freq.items():
    print(str(key) + ':' + str(val))
freq.plot(20, cumulative=False) """


ps = PorterStemmer()
""" for word in clean_tokens:
    print("{0:20}{1:20}".format(word, ps.stem(word))) """


print("\n")


lem = WordNetLemmatizer()

# for word in clean_tokens:
# print ("{0:20}{1:20}".format(word,lem.lemmatize(word)))

sentiment = []

for a in correntSentence:
    sentiment.append((a.sentiment.polarity) * 100)
print(sentiment)


def merge(correntSentence, sentiment):
    merged_list = [(correntSentence[i], sentiment[i])
                   for i in range(0, len(correntSentence))]
    return merged_list


combine = merge(correntSentence, sentiment)
# print(combine)

output = pd.DataFrame(data={"text": correntSentence, "sentiment": sentiment})

output.to_csv("CSVFormat.csv", index=False, quoting=3)
