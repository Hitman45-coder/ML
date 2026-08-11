# %% 
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)


# %%
import matplotlib.pyplot as plt
X, y = mnist['data'], mnist['target']

def plot_num(dig):
    img = dig.reshape(28,28)
    plt.imshow(img, cmap ='binary')
    plt.show()
some_digit = X[0]
plot_num(some_digit)


# %%
from sklearn.linear_model import SGDClassifier

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


# %%
from sklearn.model_selection import cross_val_predict
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv =3, method="decision_function")


# %%
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision


# %%
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))  
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")

plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)

plt.show()


# %%
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


# %%
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                     method="predict_proba")

# %%
y_probas_forest[:2]

# %%
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)
# %%
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label='Random Forest')
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.xlabel("Recall")
plt.ylabel("precision_recall_curve")
plt.axis([0 , 1, 0 , 1])
plt.show()


# %%
from sklearn.metrics import f1_score 
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5 


# %%
from sklearn.svm import SVC 

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])


# %%
svm_clf.predict([some_digit])


# %%:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))

 
# %%
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


# %% 

knn_clf.predict([some_digit])


# %%
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")


# %%
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3 , random_state = 42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])


# %%
chain_clf.predict([some_digit])


# %%
rng = np.random.default_rng(seed=42)
noise_train = rng.integers(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise_train
noise_test = rng.integers(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test
# %%
plot_num(X_train_mod[0])
 
# %%
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_num(clean_digit)
plt.show()
