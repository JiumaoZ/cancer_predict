"""
需求：
对癌症进行预测，结果有两类良性和恶性

步骤：
1. 获取数据
2. 数据预处理
    2.1 处理缺失值
    2.2 数据集划分
3. 特征工程
    3.1 标准化处理
4. 逻辑回归预估器
5. 模型评估
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# 1. 获取数据
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin','Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv(path, names=column_name)

# 2. 数据预处理
# 2.1 缺失值处理
data = data.replace(to_replace="?", value=np.nan)
data.dropna(inplace=True)
# 2.2 数据集划分
# 筛选特征值和目标值
x = data.iloc[:, 1:-1]
y = data["Class"]
x_train, x_test, y_train, y_test = train_test_split(x, y)

# 3. 特征工程
# 3.1 标准化处理
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 逻辑回归预估器
estimator = LogisticRegression()
estimator.fit(x_train, y_train)
joblib.dump(estimator, "cancer_predict_estimator.pkl")

# 5. 模型评估
# 展示模型
print("逻辑回归的回归系数是：\n", estimator.coef_)
print("逻辑回归的偏置是：\n", estimator.intercept_)
# 模型评估
y_predict = estimator.predict(x_test)
print("y_predict:\n", y_predict)
print("直接比对真实值与预测值：\n", y_test == y_predict)
score = estimator.score(x_test, y_test)
print("准确率为：\n", score)
# 查看精确率、召回率、F1-score
report = classification_report(y_test, y_predict, labels=[2, 4], target_names=["良性", "恶性"])
print(report)
# 查看ROC-AUC指标
# 将分类目标值置为0和1
y_true = np.where(y_test > 3, 1, 0)
roc_auc_score = roc_auc_score(y_true, y_predict)
print(roc_auc_score)
