import pandas as pd
import numpy as np
import seaborn as sns
import warnings as wr
import matplotlib.pyplot as plt
wr.filterwarnings("ignore")
df=pd.read_csv("C:/Users/Pranav Pawar/Downloads/ai_student_impact_dataset (1).csv")
#print(df.head())
#print(df.shape)
#print(df.info())
#print(df.describe().T)
#print(df.isnull().sum())
#print(df.duplicated().sum())
sns.boxplot(x='subject', y="value", data=df)
plt.show()
#sns.set_palette("Pastel1")
#plt.figure(figsize=(10,6))
#sns.pairplot(df)
#plt.suptitle('Pair Plot for Dataframe')
#plt.show()



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
data=pd.read_csv("C:/Users/Pranav Pawar/Downloads/train.csv")
print(data.head())

