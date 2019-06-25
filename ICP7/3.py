import urllib.request
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.stem import LancasterStemmer, SnowballStemmer, WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, ne_chunk, ngrams

# Reading Text from the URL
wikiURL = "https://en.wikipedia.org/wiki/Google"
openURL = urllib.request.urlopen(wikiURL)

# Assigning Parsed Web Page into a Variable
soup = BeautifulSoup(openURL.read(), "html.parser")

# Kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()

# get text
text = soup.body.get_text()

# break into lines and remove leading and trailing space on each
lines = (line.strip() for line in text.splitlines())

# break multi-headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# drop blank lines
text = ' '.join(i for i in chunks if i)


# Saving to a Text File
with open('Input.txt', 'w') as text_file:
    text_file.write(str(text.encode("utf-8")))


# Tokenization
######Sentence Tokenization######
sentTok = sent_tokenize(text)
print("Sentence Tokenization : \n", sentTok)

####Word Tokenization#######
tokens = [word_tokenize(t) for t in sentTok]
print ("Word Tokenization : \n", tokens)


# Stemming
#LancasterStemmer
lStem = LancasterStemmer()
print("Lancaster Stemming : \n")
for i in tokens:
    print(lStem.stem(str(i)))

#SnowBallStemmer
sStem = SnowballStemmer('english')
print("SnowBall Stemming : \n")
for i in tokens:
    print(sStem.stem(str(i)))

#PorterStemmer
pStem = PorterStemmer()
print("Porter Stemming : \n")
for i in tokens:
    print(pStem.stem(str(i)))

# POS-tagging
print("Part of Speech Tagging :\n", pos_tag(word_tokenize(text)))

# Lemmatization
lemmatizer = WordNetLemmatizer()
print("Lemmatization :\n")
for tok in tokens:
    print(lemmatizer.lemmatize(str(tok)))

# Trigram
print("Trigrams :\n")
trigram = []
for x in tokens:
    trigram.append(list(ngrams(x, 3)))
print(trigram)

# Named Entity Recognition
print("NER : \n", ne_chunk(pos_tag(wordpunct_tokenize(str(tokens)))))
