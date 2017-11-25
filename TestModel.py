import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
import numpy as np

corpus1 = []
impact1 = []

corpus2 = []

impact2 = []

range1 = 120000
range2 = 128000

clf = joblib.load('ForestClassifierModel.pkl')

# vec = CountVectorizer()
transformer = TfidfTransformer(smooth_idf=False)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8)

vectorizer = DictVectorizer(sparse=False)

with open('labeled_weather_data__vol2.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for i in range(0, range1):
        row = reader.next()
        if 'IMPACT' in row:
            if row['IMPACT'] == "2" or row['IMPACT'] == "1" or row['IMPACT'] == "-1" or row['IMPACT'] == "-2":
                corpus1.append(row)
                impact1.append(row['IMPACT'])
        impact1.append(row['IMPACT'])
        if 'IMPACT' in row:
            del row['IMPACT']
        corpus1.append(row)


    for i in range(range1, range2):
        row = reader.next()
        impact2.append(row['IMPACT'])
        # type(impact2[0])
        if 'IMPACT' in row:
            del row['IMPACT']
        corpus2.append(row)


print "Start frequency counting.."
frequencyCountTransformArray = vectorizer.fit_transform(corpus1)
print "Finish frequency counting.."

# print "Start model training.."
# clf = RandomForestClassifier(n_estimators=50, verbose=3, n_jobs=-1).fit(frequencyCountTransformArray, impact1)
# print "Model training complete, save model to ForestClassifierModel.pkl.."
# joblib.dump(clf, 'ForestClassifierModel.pkl')
# print "Model saved!"


newTfidf = vectorizer.transform(corpus2)
predicted = clf.predict(newTfidf)

for text, tag, actual in zip(corpus2, predicted, impact2):
    if tag == "2" or tag == "1" or tag == "-1" or tag == "-2":
        print('{} => {}, {}'.format(text, tag, actual))



print str(np.mean(predicted == impact2))