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


# word tokenization -- TODO: ne treba
# words =  lower_case[0].split(' ')
# print(words)

# from nltk.tokenize import word_tokenize


wordToken = []
sentenceToken = []

for a in lower_case:
    # print(a)
    sentenceToken.append(a)
    for i, word in enumerate(a.split()):
        wordToken.append(word)
        # print(wordToken)

# print(sentenceToken)
# print(wordToken)

# TODO: ne treba
# tokens = [t for t in wordToken]
# print(tokens)

# remove stopwords
sr = stopwords.words('english')
clean_tokens = wordToken[:]
for token in wordToken:
    if token in stopwords.words('english'):

        clean_tokens.remove(token)

# print(clean_tokens)

# frequency
freq = nltk.FreqDist(clean_tokens)
# for key,val in freq.items():
# print(str(key) + ':' + str(val))
# freq.plot(20, cumulative=False)

# stemming word

ps = PorterStemmer()
# for word in clean_tokens:
# print("{0:20}{1:20}".format(word,ps.stem(word)))


print("\n")
# lemmatization

lem = WordNetLemmatizer()

# for word in clean_tokens:
# print ("{0:20}{1:20}".format(word,lem.lemmatize(word)))

# print(sentenceToken)
array_length = len(sentenceToken)
obj = []
for i in range(array_length):
    # print(i)
    obj.append(TextBlob(sentenceToken[i]))

    i = i+1

# print(obj)

# detect sentences' language
foreignFeedback = []
for x in obj:
    #print(x + ' is written in: ' + x.detect_language())
    if x.detect_language() != "en":
        foreignFeedback.append(x)

""" if not foreignFeedback:
    print("There are no feedback written in foreign language (not english)")
else:
    print(foreignFeedback) """

# translate to english - TODO: ovo treba uradit

# spelling coorection - TODO: treba skontat koja ne valja i samo to da popravi
for x in obj:
    print(x)
    print(x.correct())

sentiment = []

for a in obj:
    sentiment.append((a.sentiment.polarity) * 100)

# TODO: treba bolji da bude (mogu stavit samo bolje recenice :P) - fula na I don't recommend anyone to take this car i Remove this car from the page

# print(sentiment)


def merge(obj, sentiment):
    merged_list = [(obj[i], sentiment[i]) for i in range(0, len(obj))]
    return merged_list


combine = merge(sentenceToken, sentiment)
# print(combine)

output = pd.DataFrame(data={"text": sentenceToken, "sentiment": sentiment})

# output.to_csv("CSVFormat.csv", index=False, quoting=3 )
