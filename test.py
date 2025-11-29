import pandas as pd
from sklearn.ensemble import RandomForestRegressor  
train = pd.read_csv("./dataset/fixed_train.csv")
test = pd.read_csv("./dataset/fixed_test.csv")
test_initial = pd.read_csv("./dataset/test.csv") 
y = train["Price"]
x = train.drop(columns=["Price"])
model = RandomForestRegressor(
    n_estimators=500,
    max_features="sqrt",
    random_state=0
)
model.fit(x, y)
x_test = test[x.columns]   
predicted_price = model.predict(x_test)
submission = pd.DataFrame({
    "ID": test_initial["ID"],
    "Price": predicted_price
})
submission.to_csv("./dataset/submission.csv", index=False)