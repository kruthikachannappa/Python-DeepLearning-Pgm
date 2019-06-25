from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


twenty_train = fetch_20newsgroups(categories=['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware','comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles'], shuffle=True)
stop_words = set(stopwords.words('english'))


tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = KNeighborsClassifier()

clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)


predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)