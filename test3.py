import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
train = pd.read_csv("./dataset/fixed_train.csv")
test = pd.read_csv("./dataset/fixed_test.csv")
test_initial = pd.read_csv("./dataset/test.csv") 
y = train["Price"]
x = train.drop(columns=["Price"])
gboost = GradientBoostingRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=3,
    random_state=0
)
gboost.fit(x, y)
x_test = test[x.columns]  
predicted_price = gboost.predict(x_test)
submission = pd.DataFrame({
    "ID": test_initial["ID"],
    "Price": predicted_price
})
submission.to_csv("./dataset/submission3.csv", index=False)