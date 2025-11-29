import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
train = pd.read_csv("./dataset/fixed_train.csv")
y = train["Price"]
x = train.drop(columns=["Price"])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)
tree = DecisionTreeRegressor(random_state=0)
tree.fit(x_train, y_train)
y_tree = tree.predict(x_test)
mse_tree = np.mean((y_test - y_tree)**2)
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=200,
    random_state=0
)
bagging.fit(x_train, y_train)
y_bagging = bagging.predict(x_test)
mse_bagging = np.mean((y_test - y_bagging)**2)
random_forest = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    random_state=0
)
random_forest.fit(x_train, y_train)
y_random_forest = random_forest.predict(x_test)
mse_random_forest = np.mean((y_test - y_random_forest)**2)
gboost = GradientBoostingRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=3,
    random_state=0
)
gboost.fit(x_train, y_train)
y_gboost = gboost.predict(x_test)
mse_gboost = np.mean((y_test - y_gboost)**2)
print(mse_tree)
print(mse_bagging)
print(mse_random_forest)
print(mse_gboost)