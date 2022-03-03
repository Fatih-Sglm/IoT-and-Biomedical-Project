# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:16:24 2022

@author: fatih
"""


from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


df = read_csv("project_features.csv")

sns.countplot(x="label", data=df,palette="Set3")
plt.show()

x = df.iloc[:,:6]
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

print("Train X shape:",x_train.shape)
print("Test X shape:",x_test.shape)

models=dict()
models['KNN'] = KNeighborsClassifier(n_neighbors=7)
models['DT'] = DecisionTreeClassifier()
models['SVC'] = SVC()
models['GNB'] = GaussianNB()
# ensemble models
models['BAG'] = BaggingClassifier(n_estimators=100)
models['RF'] = RandomForestClassifier(n_estimators=100)
models['ExT'] = ExtraTreesClassifier(n_estimators=100)
models['GBC'] = GradientBoostingClassifier(n_estimators=100)
print('Numbers of models: %d' % len(models))

# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
	# fit the model
	model.fit(trainX, trainy)
	# make predictions
	yhat = model.predict(testX)
	# evaluate predictions
	accuracy = accuracy_score(testy, yhat)
	return accuracy * 100.0

# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		results[name] = evaluate_model(trainX, trainy, testX, testy, model)
		# show process
		print('>%s: %.3f' % (name, results[name]))
	return results

# print and plot the results
def summarize_results(results, maximize=True):
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,v) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	print()
	for name, score in mean_scores:
		print('%s, Classification Score=%.3f' % (name, score))

# evaluate models
results = evaluate_models(x_train, y_train, x_test, y_test, models)
# summarize results
summarize_results(results)