# %%
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist['data'], mnist['target']

# %%
import matplotlib.pyplot as plt
def plot_num(some_digit):
    image = some_digit.reshape(28,28)
    plt.imshow(image, cmap= "binary")

image = X[0]
plot_num(image)
plt.show()

# %%
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train, y_train)

# %%
from sklearn.neighbors import KNeighborsClassifier

kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train_scaled, y_train)

# %%

from sklearn.model_selection import cross_val_score

acc_score = cross_val_score(kn_clf, X_train_scaled, y_train,cv =3, scoring = "accuracy")
print(acc_score)

# %%
y_pred = kn_clf.predict(X_test)

# %%
num = y_pred[34]
print(num)
# %%
plot_num(X_test[34])

