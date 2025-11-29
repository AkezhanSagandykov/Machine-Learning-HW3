import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("./dataset/train.csv", na_values=["\\N"])
test = pd.read_csv("./dataset/test.csv", na_values=["\\N"])

train.info()

print(train.describe())

print(train.isnull().sum())


quantitative = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
qualitative = train.select_dtypes(include=['object']).columns.tolist()
print(quantitative)
print(qualitative)

print(train.columns)
delete_unimportant_columns = [
    "ID", "No. of Doors", "Location",
    "Colour", 
]

train = train.drop(columns=[i for i in delete_unimportant_columns if i in train.columns])
test  = test.drop(columns=[i for i in delete_unimportant_columns if i in test.columns])

train["Owner_Type"] = train["Owner_Type"].replace(["", " "], "Not assigned")
test["Owner_Type"] = test["Owner_Type"].replace(["", " "], "Not assigned")
train["Transmission"] = train["Transmission"].replace(["", " "], "Not assigned")
test["Transmission"] = test["Transmission"].replace(["", " "], "Not assigned")
def extract_number(text):
    if isinstance(text, str):
        num = ''.join(i for i in text if (i.isdigit() or i == '.'))
        if (num):
            return float(num)
        else:
            np.nan
    return np.nan
for i in [train, test]:
    i["Mileage"] = i["Mileage"].apply(extract_number)
    i["Engine"] = i["Engine"].apply(extract_number)
    i["Power"] = i["Power"].apply(extract_number)
    i["Seats"] = pd.to_numeric(i["Seats"], errors="coerce")
def parse_price(value):
    if isinstance(value, str) and "Lakh" in value:
        return float(value.split()[0])
    return np.nan

for i in [train, test]:
    i["New_Price"] = i["New_Price"].apply(parse_price)

quantitative_columns = train.select_dtypes(include=[np.number]).columns

for i in quantitative_columns:
    average_value = train[i].mean()

    train[i] = train[i].fillna(average_value)

    if i in test.columns:
        test[i] = test[i].fillna(average_value)
qualititative_columns = train.select_dtypes(include=["object"]).columns

for i in qualititative_columns:
    train[i] = train[i].astype(str)
    test[i] = test[i].astype(str)

    train_distinct_values = set(train[i].unique())

    test[i] = test[i].apply(lambda x: x if x in train_distinct_values else "Other values")
    train[i] = train[i].apply(lambda x: x if x in train_distinct_values else "Other values")

    label_encoder = LabelEncoder()
    label_encoder.fit(list(train_distinct_values) + ["Other values"])

    train[i] = label_encoder.transform(train[i])
    test[i] = label_encoder.transform(test[i])

train.to_csv("./dataset/fixed_train.csv", index=False)
test.to_csv("./dataset/fixed_test.csv", index=False)