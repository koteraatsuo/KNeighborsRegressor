
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# アヤメデータセット読み込み
from sklearn.datasets import load_iris
iris = load_iris()

# 特徴量
X = iris.data
# 目的変数
Y = iris.target

# データ表示（特徴量）
print("データ数 = %d  特徴量 = %d" % (X.shape[0], X.shape[1]))
pd.DataFrame(X, columns=iris.feature_names).head()

# データ表示（目的変数）
print("データ数 = %d" % (Y.shape[0]))
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

#
# K-近傍法
#
from sklearn.neighbors import KNeighborsClassifier

list_nn = []
list_score = []
for k in range(1, 31): # K = 1~30
  # KNeighborsClassifier
  knc = KNeighborsClassifier(n_neighbors=k)
  knc.fit(X_train, Y_train)

  # 予測　
  Y_pred = knc.predict(X_test)

  # 評価 R^2
  score = knc.score(X_test, Y_test)
  print("[%d] score: {:.2f}".format(score) % k)

  list_nn.append(k)
  list_score.append(score)

# プロット
plt.ylim(0.9, 1.0)
plt.xlabel("n_neighbors")
plt.ylabel("score")
plt.plot(list_nn, list_score)