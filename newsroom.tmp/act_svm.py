from sklearn.datasets import fetch_20newsgroups
from sklearn import cross_validation
import numpy as np

#cats = ['alt.atheism', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',#categories=cats,
                                    remove=('headers','footers','quotes'));
from pprint import pprint
pprint(list(newsgroups_train.target_names));

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer();
vectors = vectorizer.fit_transform(newsgroups_train.data);
print(vectors.shape);

from sklearn.svm import LinearSVC
from sklearn import metrics
clf = LinearSVC();
randomErr = [];
randomTrainX = [];
randomTrainY = [];

X_train, X_test, y_train, y_test = cross_validation.train_test_split(vectors, newsgroups_train.target,test_size=0.9)
def getRandom():
    point = np.random.rand()*X_train.shape[0];
    point = int(point);
    y = y_train[point];
    X = X_train[point];
    return X,y;

def ClosestToLine(hyperplane,points,intercept,number):
    """Calculates distance vector from points to hyperplane"""
    print(type(points.data));
    print(size(points.data));
    dist = abs(np.matmul(points,hyperplane)+intercept);
    return dist.argsort()[:number];

newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers','footers',
                                            'quotes'));#categories=cats);
vectors_test = vectorizer.transform(newsgroups_test.data);

randomTrainX,randomTrainY = X_train[:10],y_train[:10]

i=0;
while i < 300:
    #X,y = getRandom();
    randomTrainX = X_train[:i+10];
    randomTrainY = y_train[:i+10];
    #randomTrainvectorsX = vectorizer.transform(randomTrainX);
    clf.fit(randomTrainX, randomTrainY);
    pred = clf.predict(vectors_test);

    randomErr.append(metrics.f1_score(newsgroups_test.target, pred,
                        average='macro'));
    i=i+10;
    print("Iteration :"+str(i)+"\t\tError :"+str(randomErr[-1]));
    print(clf.intercept_[0]);

i=0;
ClosestTrainX,ClosestTrainY = X_train[:10],y_train[:10]
closestErr=[];
print(type(X_train));
print(X_train[0].shape);
print(clf.coef_[0])
while i<300:
        points=ClosestToLine(clf.coef_[0],X_train,clf.intercept_[0],10);
        print(points);
        i += 10;

import matplotlib.pyplot as plt
plt.plot(randomErr);
plt.show();

#import numpy as np
#def show_top10(classifer,vectorizer,categories):
#    feature_names = np.asarray(vectorizer.get_feature_names())
#    for i, category in enumerate(categories):
#        top10 = np.argsort(classifer.coef_[i])[-10:]
#        print("%s: %s" %(category, " ".join(feature_names[top10])))
#
#show_top10(clf, vectorizer, newsgroups_train.target_names);
