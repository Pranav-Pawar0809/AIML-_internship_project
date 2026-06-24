
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

data = pd.read_csv("C:/Users/ Pranav Pawar/Downloads/train.csv")
print(data.head())
data.columns
y = data['price_range']
X = data.drop('price_range', axis=1)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=7)
model = RandomForestClassifier(random_state=7, n_estimators=100)
model.fit(train_X, train_y)
