import pandas as pd
from sklearn.tree import  DecisionTreeRegressor
train = pd.read_csv("./dataset/fixed_train.csv")
test = pd.read_csv("./dataset/fixed_test.csv")
test_initial = pd.read_csv("./dataset/test.csv") 
y = train["Price"]
x = train.drop(columns=["Price"])
tree = DecisionTreeRegressor(random_state=0)
tree.fit(x, y)
x_test = test[x.columns]   
predicted_price = tree.predict(x_test)
submission = pd.DataFrame({
    "ID": test_initial["ID"],
    "Price": predicted_price
})
submission.to_csv("./dataset/submission1.csv", index=False)