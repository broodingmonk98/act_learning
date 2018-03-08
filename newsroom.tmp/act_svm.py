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
##nnz data indices indptr has_sortedindices
print(vectors.shape);

from sklearn.svm import LinearSVC
from sklearn import metrics
clf = LinearSVC();
randomErr = [];
randomTrainX = [];
randomTrainY = [];

X_train, X_test, y_train, y_test = cross_validation.train_test_split(vectors, newsgroups_train.target,test_size=0.2)
def getRandom():
    point = np.random.rand()*X_train.shape[0];
    point = int(point);
    y = y_train[point];
    X = X_train[point];
    return X,y;

def ClosestToLine(hyperplane,points,intercept,number,alreadyThere):
    """Calculates distance vector from points to hyperplane"""
##nnz data indices indptr has_sortedindices
    dist = abs(points._mul_vector(hyperplane)+intercept);
    mask = np.zeros(dist.shape, dtype=bool);
    mask[alreadyThere] = True;
    dist[mask] = float('-inf');
    return dist.argsort()[-number:];

newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers','footers',
                                            'quotes'));#categories=cats);
vectors_test = vectorizer.transform(newsgroups_test.data);

randomTrainX,randomTrainY = X_train[:10],y_train[:10]

i=0;
while i < 2500:
    #X,y = getRandom();
    randomTrainX = X_train[:i+50];
    randomTrainY = y_train[:i+50];
    #randomTrainvectorsX = vectorizer.transform(randomTrainX);
    clf.fit(randomTrainX, randomTrainY);
    pred = clf.predict(vectors_test);

    randomErr.append(metrics.f1_score(newsgroups_test.target, pred,
                        average='macro'));
    i=i+50;
    print("Iteration :"+str(i)+"\t\tError :"+str(randomErr[-1]));

i=0;
ClosestTrainX,ClosestTrainY = X_train[:10],y_train[:10]
closestErr=[];
clf.fit(ClosestTrainX,ClosestTrainY);
#print(type(X_train));
#print(X_traiXn[0].shape);
print("\nSmart select : \n\n\n\n\n");
#print(clf.coXef_[0])
points = np.array([0,1,2,3,4,5,6,7,8,9])
while i<2500:
    points =np.concatenate((points,ClosestToLine(clf.coef_[0],X_train,clf.intercept_[0],50,points)));
    print((points));
    ClosestTrainX=X_train[points]
    ClosestTrainY=y_train[points]
    clf.fit(ClosestTrainX,ClosestTrainY);
    pred = clf.predict(vectors_test);
    closestErr.append(metrics.f1_score(newsgroups_test.target, pred,
                    average='macro'));
    i += 50;
    print("Iteration :"+str(i)+"\t\tError :"+str(closestErr[-1]));
    print("Unique = "+str(len(np.unique(points))));

import matplotlib.pyplot as plt
line1 = plt.plot(randomErr,label='Random');
line2 = plt.plot(closestErr,label='Smart search');
#plt.legend((line1,line2),('random','smart'))
plt.show();

#import numpy as np
#def show_top10(classifer,vectorizer,categories):
#    feature_names = np.asarray(vectorizer.get_feature_names())
#    for i, category in enumerate(categories):
#        top10 = np.argsort(classifer.coef_[i])[-10:]
#        print("%s: %s" %(category, " ".join(feature_names[top10])))
#
#show_top10(clf, vectorizer, newsgroups_train.target_names);
