import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer

corpus = []
impact = []

range1 = 120000
range2 = 128000

# vec = CountVectorizer()
# transformer = TfidfTransformer(smooth_idf=False)
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.8)

vectorizer = DictVectorizer(sparse=False)

with open('labeled_weather_data__vol2.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for i in range(0, range1):
        row = reader.next()
        if 'IMPACT' in row:
            if row['IMPACT'] == "2" or row['IMPACT'] == "1" or row['IMPACT'] == "-1" or row['IMPACT'] == "-2":
                corpus.append(row)
                impact.append(row['IMPACT'])
        impact.append(row['IMPACT'])
        if 'IMPACT' in row:
            del row['IMPACT']
        corpus.append(row)


print "Start frequency counting.."
frequencyCountTransformArray = vectorizer.fit_transform(corpus)
print "Finish frequency counting.."

print "Start model training.."
clf = RandomForestClassifier(n_estimators=100, verbose=3, n_jobs=-1).fit(frequencyCountTransformArray, impact)
print "Model training complete, save model to ForestClassifierModel.pkl.."
joblib.dump(clf, 'ForestClassifierModel.pkl')
print "Model saved!"