import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv(r"D:\model_with_flask\dataset.csv")

# Separate features (X) and target variable (y)
X = df.iloc[:, 0:10].values
y = df['Fake']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Evaluate the model
print("TRAIN SET", clf.score(X_train, y_train))
print("TEST  SET", clf.score(X_test, y_test))

# Generate classification report
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model to a file in PKL format
import pickle
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)
