from sklearn.datasets import fetch_20newsgroups
from sklearn import preprocessing, cross_validation,svm
#cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',#categories=cats,
                                    remove=('headers','footers','quotes'));
from pprint import pprint
pprint(list(newsgroups_train.target_names));

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer();
vectors = vectorizer.fit_transform(newsgroups_train.data);
X_train, X_test, y_train, y_test = cross_validation.train_test_split(vectors, newsgroups_train.target,test_size=0.6)
print(X_train.shape);

from sklearn.svm import LinearSVC
from sklearn import metrics
clf = LinearSVC();
clf.fit(X_train, y_train)

newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers','footers',
                                            'quotes'))#,categories=cats);
vectors_test = vectorizer.transform(newsgroups_test.data);

pred = clf.predict(vectors_test);
print(metrics.f1_score(newsgroups_test.target, pred, average='macro'));

#import numpy as np
#def show_top10(classifer,vectorizer,categories):
#    feature_names = np.asarray(vectorizer.get_feature_names())
#    for i, category in enumerate(categories):
#        top10 = np.argsort(classifer.coef_[i])[-10:]
#        print("%s: %s" %(category, " ".join(feature_names[top10])))
#
#show_top10(clf, vectorizer, newsgroups_train.target_names);
