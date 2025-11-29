import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
train = pd.read_csv("./dataset/fixed_train.csv")
test = pd.read_csv("./dataset/fixed_test.csv")
test_initial = pd.read_csv("./dataset/test.csv") 
y = train["Price"]
x = train.drop(columns=["Price"])
bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=500,
    random_state=0
)
bagging.fit(x, y)
x_test = test[x.columns]  
predicted_price = bagging.predict(x_test)
submission = pd.DataFrame({
    "ID": test_initial["ID"],
    "Price": predicted_price
})
submission.to_csv("./dataset/submission2.csv", index=False)