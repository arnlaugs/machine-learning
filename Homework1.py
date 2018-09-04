import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


n=100
x = np.random.rand(n,1)
y = 5*x*x+0.1*np.random.randn(n,1)

xb = np.c_[np.ones((n,1)), x, x**2]
beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
#xnew = np.array([[0],[2]])
xbnew = np.c_[np.ones((n,1)), x, x**2]
ypredict = xbnew.dot(beta)

X = x[:, np.newaxis]



model = make_pipeline(PolynomialFeatures(2))
model.fit(X, y)
y_plot = model.predict(X)



#print(xb)

#print(xbnew)
plt.plot(x,y,'x')
plt.plot(x,ypredict,'o')
plt.plot(X,y_plot, '+')
plt.show()