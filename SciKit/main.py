import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.externals import joblib

#Try and skip bad lines
dframe = pd.read_csv("ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)

#Drop null values
dframe.dropna(inplace=True)

#Check if null values left
#print dframe[dframe.isnull().any(axis=1)].size

#First 5k rows
dframe =  dframe[:10000]
#Ignore columns, tag is prediction
x_df = dframe.drop(['Unnamed: 0', 'sentence_idx', 'tag'], axis=1)
#print x_df.head()



vectorizer = DictVectorizer(sparse=False)
x = vectorizer.fit_transform(x_df.to_dict("records"))

#print x.shape


#The output class
y = dframe.tag.values
all_classes = np.unique(y)
#print all_classes.shape
#print y.shape


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


print(x_train.shape)
print(y_train.shape)




clf = Perceptron(verbose=10, n_jobs=-1, n_iter=5)
all_classes = list(set(y))
clf.partial_fit(x_train, y_train, all_classes)



clf = joblib.dump(clf, 'clf.model')
print "Done"

clf = joblib.load('clf.model')

print(f1_score(clf.predict(x_test), y_test, average="micro"))