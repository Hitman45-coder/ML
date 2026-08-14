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
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
X_train_normalized =  X_train / 255
X_test_normalized = X_test / 255


# %%
from sklearn.neighbors import KNeighborsClassifier

kn_clf = KNeighborsClassifier()
# kn_clf.fit(X_train_scaled, y_train)


# %%

from sklearn.model_selection import cross_val_score

# acc_score = cross_val_score(kn_clf, X_train_scaled, y_train,cv =3, scoring = "accuracy")
# print(acc_score)

# %%
# y_pred = kn_clf.predict(X_test)

# %%
# num = y_pred[34]
# print(num)
# %%
plot_num(X_test[34])


# %%
# acc = cross_val_score(kn_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
# print(acc)


# %%
param_grid = [
        {   
            'n_neighbors': [3, 4, 5] ,
            'weights': ['uniform', 'distance'],
        }
    ] 


# %%
from sklearn.model_selection import GridSearchCV

# grid_search = GridSearchCV(kn_clf, param_grid= param_grid, cv=3, scoring='accuracy', n_jobs= -1)
# grid_search.fit(X_train, y_train)

# print("Best parameters found:", grid_search.best_params_)
# print("Best cross-validation:", grid_search.best_score_)


# %%
from scipy.ndimage import shift

def shift_image(image: np.array):
    image = image.reshape(28, 28)
    return (    
                (shift(image, (0,0))).reshape(784),  # original
                (shift(image, (-1,0))).reshape(784), # top
                (shift(image, (0,1))).reshape(784),  # right
                (shift(image, (1,0))).reshape(784),  # bottom
                (shift(image, (0,-1))).reshape(784)   # left
            )

def extend_y(y: np.array):
    return (
                y,
                y,
                y,
                y,
                y
            )

# %%
print(type(y_train[0]))
# %%
X_train_backup = np.array(list(map(shift_image, X_train)))
y_train_backup = np.array(extend_y(y_train))


# %%
X_train_backup2 = X_train_backup.reshape(300000, 784)

# %%
X_train_backup.shape

# %%
X_train.shape

# %%
X_train_backup2.shape
# %%
X_train[0].shape
# %%
X_train_backup2[0].shape


# %%
y_train_backup.shape
# %%
y_train_backup2 = y_train_backup.reshape(300000)
y_train_backup2.shape
# %%
grid_search = GridSearchCV(kn_clf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train_backup2, y_train_backup2)

print("Best estimator found:", grid_search.best_params_)
print("Best cross val found:", grid_search.best_score_)
