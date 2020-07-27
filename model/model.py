import pandas as pd
import pickle

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


# Dataset loading
iris = load_iris() 

target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

features = df.iloc[:, 0:4].values
target = df['target'].values

# Divide in data training and data test
X_train, X_test, y_train, y_test = train_test_split(features,
													target,
													test_size=0.2,
													random_state=9)

# Simple tree
tree = DecisionTreeClassifier(	max_depth=1,
								splitter='random',
								max_features='sqrt')

# Multi tree
trees = BaggingClassifier(	base_estimator=tree,
							n_estimators=1000,
							bootstrap=False)

# Model training
model = trees.fit(X_train, y_train)
# Model predict
pred = model.predict(X_test)
# Model performance
score = accuracy_score(y_test, pred)

# Saving trained model
with open('model.pkl', 'wb') as file:
	pickle.dump(model, file)
