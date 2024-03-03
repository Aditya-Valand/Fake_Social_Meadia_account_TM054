# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import classification_report

# # Load the dataset
# df = pd.read_csv(r"D:\model_with_flask\dataset.csv")

# # Separate features (X) and target variable (y)
# X = df.iloc[:, 0:10].values
# y = df['Fake']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Initialize and train the MLP classifier
# clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0)
# clf.fit(X_train, y_train)

# # Evaluate the model
# print("TRAIN SET", clf.score(X_train, y_train))
# print("TEST  SET", clf.score(X_test, y_test))

# # Generate classification report
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Save the trained model to a file in PKL format
# import pickle
# with open('mlp_model.pkl', 'wb') as model_file:
#     pickle.dump(clf, model_file)
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv(r"D:\model_with_flask\dataset.csv")

# Separate features (X) and target variable (y)
X = df.iloc[:, 0:10].values
y = df['Fake']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

# Initialize and train the MLP classifier using GridSearchCV for hyperparameter tuning
clf = GridSearchCV(MLPClassifier(max_iter=1000, random_state=0), param_grid, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate the model
print("Best parameters found:", clf.best_params_)
print("TRAIN SET", clf.best_estimator_.score(X_train, y_train))
print("TEST  SET", clf.best_estimator_.score(X_test, y_test))

# Generate classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model to a file in PKL format
import pickle
with open('mlp_model1.pkl', 'wb') as model_file:
    pickle.dump(clf.best_estimator_, model_file)
