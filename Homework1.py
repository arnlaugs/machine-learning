import numpy as np
from random import random, seed
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


n=100
x = np.random.rand(n,1)


y = 5*x*x+0.01*np.random.randn(n,1)



xb = np.c_[np.ones((n,1)), x, x**2]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
xbnew = np.c_[np.ones((n,1)), x, x**2]
ypredict = xbnew.dot(beta)


poly2 = PolynomialFeatures(degree=2)

X = poly2.fit_transform(x)

clf2 = LinearRegression()
clf2.fit(X,y)



plt.plot(x,y,'x')
plt.plot(x,ypredict,'o')
plt.plot(x,clf2.predict(X),'+')

plt.show()

mse=mean_squared_error(y,clf2.predict(X))
print(mse)