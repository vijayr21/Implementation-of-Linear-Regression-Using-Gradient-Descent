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
Developed by:VIJAY R
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

![Screenshot 2024-08-28 094001](https://github.com/user-attachments/assets/1699423c-0d72-4923-b3b5-1f0b05eb92d7)


Compute cost value:

![Screenshot 2024-08-28 094010](https://github.com/user-attachments/assets/6a6dae5c-2924-404c-831f-0c6ddc83d58a)


h(x) Value:

![Screenshot 2024-08-28 094017](https://github.com/user-attachments/assets/9a4d2104-3991-4162-b73b-f0378c2d9ad0)


Plt.profitprediction:

![image](https://github.com/user-attachments/assets/935ec655-9759-4fe4-a196-5d7431499938)


Cost function using Gradient Descent:

![image](https://github.com/user-attachments/assets/06e0f89e-0734-45fe-a410-a579840392e9)


Profit Prediction:

![image](https://github.com/user-attachments/assets/ad3119e9-2c1a-4d8d-a5cd-09caf2ff916d)


predict1:

![image](https://github.com/user-attachments/assets/3fb40a75-7809-453d-9364-7f6eb3180634)


predict2:

![image](https://github.com/user-attachments/assets/bed7f45f-666c-4dbe-b8b9-6fa4ba2fb3a8)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
