

import sys
sys.path.append('../functions')

from functions import FrankeFunction, MSE, R2_Score

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed


def linear(X,z):

    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_)

    zpredict_ = X.dot(beta)
    #Reshape to a matrix
    return beta,np.reshape(zpredict_,z.shape)

    
    
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#Transform from matricies to vectors
x_=np.ravel(x)
y_=np.ravel(y)
z_=np.ravel(z)
n=len(x_)
i=np.random.randint(n-1, size=int(n*0.2))
x_learn=np.delete(x_,i)
y_learn=np.delete(y_,i)
x_test=np.take(x_,i)
y_test=np.take(y_,i) 

#5th order 
X=np.c_[np.ones((n,1)),x_,y_,x_*x_,y_*x_,y_*y_,x_**3,x_**2*y_,x_*y_**2,y_**3,
        x_**4,x_**3*y_,x_**2*y_**2,x_*y_**3,y_**4, 
        x_**5,x_**4*y_,x_**3*y_**2,x_**2*y_**3,x_*y_**4,y_**5]


beta,zpredict=linear(X,z)

MSE_=MSE(z,zpredict)
print(R2_Score(z,zpredict))
print(MSE_)







fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x,y,zpredict)#,cmap=cm.coolwarm)



# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

#plt.show()
