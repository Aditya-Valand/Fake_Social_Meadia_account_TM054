import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset
df = pd.read_csv("D:\model_with_flask\dataset.csv")

# Extracting features (X) and target (y)
X = df.iloc[:, 0:10].values
y = df['Fake'].values

# Normalize the features
from sklearn import preprocessing
X = preprocessing.normalize(X)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Evaluate the model
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
print("TRAIN SET accuracy:", train_accuracy)
print("TEST  SET accuracy:", test_accuracy)

# Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, y, cv=4)
print("Average cross-validation score:", scores.mean())

# Save the trained model to a file in PKL format
with open('decision_tree_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# To load the model later:
# with open('decision_tree_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
