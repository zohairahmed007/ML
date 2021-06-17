import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("carprices.csv")
df.head()
df.shape


plt.xlabel("Mileage")
plt.ylabel("Sell Price($)")
plt.title("First Corelations")
plt.scatter(df["Mileage"],df["Sell Price($)"])


plt.xlabel("Age(yrs)")
plt.ylabel("Sell Price($)")
plt.title("First Corelations")
plt.scatter(df["Age(yrs)"],df["Sell Price($)"])


X=df[["Mileage","Age(yrs)"]]
y=df["Sell Price($)"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3)

from sklearn.linear_model import LinearRegression

lr_mdl=LinearRegression()

lr_mdl.fit(X_train, y_train)

y_prd=lr_mdl.predict(X_test)


print(lr_mdl.score(X_test,y_test))

from sklearn import metrics
print('Mean squared logarithmic error:', metrics.mean_squared_log_error(y_test, y_prd))


 