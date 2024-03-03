import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pickle 

# Load the dataset
df = pd.read_csv(r"D:\model_with_flask\dataset.csv")

# Splitting the dataset into features (X) and target (y)
X = df.iloc[:, 0:10].values
y = df['Fake'].values

# Normalize the features
from sklearn import preprocessing
X = preprocessing.normalize(X)

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create and train the SVC model
clf_svm = SVC(gamma='auto', probability=True)
clf_svm.fit(X_train, y_train)

# Evaluate the model
train_accuracy = clf_svm.score(X_train, y_train)
test_accuracy = clf_svm.score(X_test, y_test)
print("TRAIN SET accuracy:", train_accuracy)
print("TEST  SET accuracy:", test_accuracy)

# Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf_svm, X, y, cv=4)
print("Average cross-validation score:", scores.mean())

# Confusion matrix
y_pred = clf_svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

# Classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# ROC curve
def plot_roc_curve(y_test, y_pred_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

y_pred_prob = clf_svm.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
plot_roc_curve(y_test, y_pred_prob)

# Save the trained model to a file
with open('svm1.pkl', 'wb') as model_file:
    pickle.dump(clf_svm, model_file)

# Load the model from the pickle file
with open('svm1.pkl', 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)
