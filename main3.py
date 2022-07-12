# ボストン住宅価格データセット
# 乳癌データセット
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn import datasets
boston = datasets.load_boston()

# 説明変数
X = boston.data
# 目的変数
Y = boston.target

# データ表示（特徴量）
print("データ数 = %d  特徴量 = %d" % (X.shape[0], X.shape[1]))
pd.DataFrame(X, columns=boston.feature_names).head()

# データ表示（目的変数）
print("データ数 = %d" % (Y.shape[0]))
print(Y[:10]) # 先頭 10件表示


# 説明変数に部屋数のみ使用
X = boston.data[:, [5]] # 部屋数

# トレーニング・テストデータ分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# プロット
plt.xlabel("RM")
plt.ylabel("Target")
plt.plot(X_train, Y_train, "o")


plt.show()



from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

list_k = []
list_score = []
for k in range(1, 21):
  # KNeighborsClassifier
  knr = KNeighborsRegressor(n_neighbors=k)
  knr.fit(X_train, Y_train)

  # 予測　
  Y_pred = knr.predict(X_test)

  #
  # 評価
  #
  # 平均絶対誤差(MAE)
  mae = mean_absolute_error(Y_test, Y_pred)
  # 平方根平均二乗誤差（RMSE）
  rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))  
  # スコア R^2
  score = knr.score(X_test, Y_test)

  print("[%d] MAE = %.2f,  RMSE = %.2f,  score = %.2f" % (k, mae, rmse, score))

  list_k.append(k)
  list_score.append(score)

# プロット
plt.ylim(0, 0.7)
plt.xlabel("k")
plt.ylabel("score")
plt.plot(list_k, list_score)


# # テストデータ上での正解値（青）と予測値（赤）をプロット
# K6_Pred = np.array(list_pred)[5]

# plt.xlabel("RM")
# plt.ylabel("Target")
# plt.plot(X_test, K6_Pred, "ro")
# plt.plot(X_test, Y_test, "o")



plt.show()

