""" from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import model_selection """
from textblob import TextBlob
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

# TODO: mozda obrisar sve sto nije tekst i onda uradit lowercase

# Lower case
letters_only = re.sub("[^a-zA-Z]", " ", text2)
letters_only = letters_only.replace("   ", "\n")
lines = letters_only.splitlines()
# print(lines)

#print(" " . join(letters_only))
# print(letters_only.strip())

lower_case = [x.lower() for x in lines]
# print(lower_case)


# word tokenization -- TODO: ne treba
#words =  lower_case[0].split(' ')
# print(words)

#from nltk.tokenize import word_tokenize


wordToken = []
sentenceToken = []

# TODO: ima visak red...treba popravit

for a in lower_case:
    # print(a)
    sentenceToken.append(a)
    for i, word in enumerate(a.split()):
        wordToken.append(word)
        # print(wordToken)

# print(sentenceToken)
# print(wordToken)

# TODO: ne treba
#tokens = [t for t in wordToken]
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
#print(str(key) + ':' + str(val))
#freq.plot(20, cumulative=False)

# stemming word

ps = PorterStemmer()
# for word in clean_tokens:
# print("{0:20}{1:20}".format(word,ps.stem(word)))


print("\n")
# lemmatization

lem = WordNetLemmatizer()

# for word in clean_tokens:
#print ("{0:20}{1:20}".format(word,lem.lemmatize(word)))

# print(sentenceToken)
array_length = len(sentenceToken)
obj = []
for i in range(array_length):
    # print(i)
    obj.append(TextBlob(sentenceToken[i]))

    i = i+1

# TODO: treba obrisat whitespace

# print(obj)

# detect sentences' language
# print(obj[0].detect_language())

sentiment = []

for a in obj:
    sentiment.append((a.sentiment.polarity) * 100)

# TODO: treba bolji da bude (mogu stavit samo bolje recenice :P)

# print(sentiment)


def merge(obj, sentiment):
    merged_list = [(obj[i], sentiment[i]) for i in range(0, len(obj))]
    return merged_list


combine = merge(sentenceToken, sentiment)
print(combine)

output = pd.DataFrame(data={"text": sentenceToken, "sentiment": sentiment})

#output.to_csv("CSVFormat.csv", index=False, quoting=3 )
