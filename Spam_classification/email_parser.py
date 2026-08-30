import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

ham_dir = "data/ham/" 
spam_dir = "data/spam/"

def extract_file_names(spam_dir: str, ham_dir: str):
    """Extract the file names from the data folder."""
    spam_file_names = None 
    non_spam_file_names = None 
    for (root, dirs, files) in os.walk(ham_dir, topdown=True):
        non_spam_file_names = files

    for (root, dirs, files) in os.walk(spam_dir, topdown=True):
        spam_file_names = files
        
    return spam_file_names, non_spam_file_names


def extract_file_raw(file_name: str, content: list):
    """Extract the raw content from file."""

    file = open(file_name, 'r', errors='ignore')
    content.append(file.read())
    return content

def extract_file_processed(ham_dir: str, spam_dir: str,lowercase= True,
                           punctuation= True, stemming= False,
                           convert_to_generic= False):

    """Preprocess the raw text"""

    #ToDo: use the parameters passed
    spam_file_content = []
    ham_file_content = []
    spam_file_names, ham_file_names = extract_file_names(spam_dir, ham_dir)

    for i in range(len(spam_file_names)):
        extract_file_raw(os.path.join(spam_dir, spam_file_names[i]), spam_file_content )
    for i in range(len(ham_file_names)):
        extract_file_raw(os.path.join(ham_dir, ham_file_names[i]), ham_file_content)

    return spam_file_content, ham_file_content

# Create the whole dataset
spam_content, ham_content = extract_file_processed(ham_dir=ham_dir, spam_dir=spam_dir)
X_raw = spam_content + ham_content
y = np.array([1] * len(spam_content) + [0] * len(ham_content))
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42 )


# ------------------------ KNN classifier -----------------------------

knn_params = {
    'knn__n_neighbors': [3, 4, 5, 10],
    'knn__weights' : ['uniform', 'distance'],
}

knn_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words="english",decode_error='ignore',min_df=2)),
    ('knn',KNeighborsClassifier()) 
])
cv_strat =StratifiedKFold(n_splits=10, shuffle=True, random_state=45)

knn_grid_search = GridSearchCV(estimator=knn_pipeline, param_grid=knn_params, scoring='roc_auc_ovr', cv=cv_strat)
knn_grid_search.fit(X_train_raw,y_train)


best_model = knn_grid_search.best_estimator_

# Get predictions
y_pred = best_model.predict(X_test_raw)
y_proba = best_model.predict_proba(X_test_raw)[:, 1]

print("Best parameters found:", knn_grid_search.best_knn_params_)
print("Best CV ROC-AUC score:", knn_grid_search.best_score_)
print("---------------------------------------")
print("Test ROC-AUC Score:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ------------------------ Linar Regrssion classifier -----------------------------

reg_params= {
        'reg__C': [ 0.1, 1.0 , 10.0]
}

reg_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words="english",decode_error='ignore',min_df=2)),
    ('reg',LogisticRegression()) 
])

reg_grid_search = GridSearchCV(estimator=reg_pipeline, param_grid=reg_params, scoring='roc_auc_ovr', cv=cv_strat)
reg_grid_search.fit(X_train_raw,y_train)


best_reg_model = reg_grid_search.best_estimator_

# Get predictions
y_pred = best_reg_model.predict(X_test_raw)
y_proba = best_reg_model.predict_proba(X_test_raw)[:,1]

print("Best parameters found:", reg_grid_search.best_params_)
print("Best CV ROC-AUC score:", reg_grid_search.best_score_)
print("----------------------------")
print("\n--- Test Set Performance---")
print("Test ROC-AUC Score:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))