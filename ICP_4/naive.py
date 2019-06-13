import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('./glass.csv')
y = df["Type"]
df1 = df.drop("Type",axis=1).copy()
# create training and testing
X_train, X_test, Y_train, Y_test = train_test_split(df1, y,test_size=0.15)

model = GaussianNB()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
acc_svc = round(model.score(X_test, Y_test) * 100, 2)
print("Naive Byes accuracy with test is:", acc_svc)
plt.plot(Y_test,label="Y_test")
plt.plot(Y_pred,label="Y_pred")
plt.legend()
plt.show()