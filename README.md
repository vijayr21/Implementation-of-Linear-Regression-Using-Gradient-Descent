# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.import numpy as np.

3.Give the header to the data.

4.Find the profit of population.

5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.

6.End the program. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VIJAY R
RegisterNumber:  212223240178
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("ex1.txt",header=None)
data

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history

theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000s")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
DATASET:

![image](https://github.com/user-attachments/assets/ebae57e6-a85d-4593-bfaa-84737d4bb3f4)



Compute cost value:

![image](https://github.com/user-attachments/assets/443e6934-f97b-47a4-823d-4f3897365de3)


h(x) Value:

![image](https://github.com/user-attachments/assets/7fd5f0e2-9e6e-4d4b-89b0-8887ebd0188b)


Plt.profitprediction:

![image](https://github.com/user-attachments/assets/33299712-b8c6-422e-b219-fd7bfc37d466)



Cost function using Gradient Descent:

![image](https://github.com/user-attachments/assets/a9384956-5feb-4ea1-aae1-2a8fcedd0496)



Profit Prediction:

![image](https://github.com/user-attachments/assets/5f18b5b1-f641-4d96-8a3d-1d53081bca2a)



predict1:

![image](https://github.com/user-attachments/assets/0d0f1cf2-93c8-4930-8220-7dc88b685ec8)


predict2:

![image](https://github.com/user-attachments/assets/d431716a-d1a0-498b-af3a-b3fbe4d17f55)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
