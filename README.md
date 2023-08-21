# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neurons are the basic input/output units found in neural networks. These units are connected to one another, and each connection carries a weight. Because they are adaptable, neural networks can be applied to both classification and regression. We'll examine how neural networks can be used to tackle regression issues in this post.

A relationship between a dependent variable and one or more independent variables can be established with the aid of regression. Only when the regression equation is a good fit for the data can regression models perform well. Although sophisticated and computationally expensive, neural networks are adaptable and can choose the optimum form of regression dynamically. If that isn't sufficient, hidden layers can be added to enhance prediction. Create your training and test sets using the dataset; in this case, we are creating a neural network with a second hidden layer that uses the activation layer as relu and contains its nodes. We will now fit our dataset before making a value prediction.

## Neural Network Model

![image](https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/76f3fea5-0093-4e48-8034-c7c005aea3d3)


## DESIGN STEPS

<b>STEP 1:</b> Loading the dataset.

<b>STEP 2:</b> Split the dataset into training and testing.

<b>STEP 3:</b> Create MinMaxScalar objects ,fit the model and transform the data.

<b>STEP 4:</b> Build the Neural Network Model and compile the model.

<b>STEP 5:</b> Train the model with the training data.

<b>STEP 6:</b> Plot the performance plot.

<b>STEP 7:</b> Evaluate the model with the testing data.

## PROGRAM:
```
Developed By: D.Amarnath Redddy
RegNo: 212221240012
```
## Importing Required Packages :
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
```
## Authentication and Creating DataFrame From DataSheet :
```
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl').sheet1
data = worksheet.get_all_values()
dataset = pd.DataFrame(data[1:], columns=data[0])
dataset = dataset.astype({'Input':'float'})
dataset = dataset.astype({'Output':'float'})
dataset.head()
```
## Assigning X and Y values :
```
X = dataset[['Input']].values
Y = dataset[['Output']].values
```
## Normalizing the data :
```
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state = 20)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train_scale = Scaler.transform(x_train)
```
## Creating and Training the model :
```
my_brain = Sequential([
    Dense(units = 4, activation = 'relu' , input_shape=[1]),
    Dense(units = 6),
    Dense(units = 1)

])
my_brain.compile(optimizer='rmsprop',loss='mse')
my_brain.fit(x=x_train_scale,y=y_train,epochs=20000)
```
## Plot the loss :
```
loss_df = pd.DataFrame(my_brain.history.history)
loss_df.plot()
```
## Evaluate the Model :
```
x_test1 = Scaler.transform(x_test)
my_brain.evaluate(x_test1,y_test)
```
## Prediction for a value :
```
X_n1 = [[30]]
input_scaled = Scaler.transform(X_n1)
my_brain.predict(input_scaled)
```
## Dataset Information

<img width="236" alt="sheet" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/3a305b8f-776f-4f69-9608-715dad386d8d">



## OUTPUT:
## Input & Output Data :
<img width="187" alt="11" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/25fc4418-fae5-4560-946b-f7a39713b834">


### Training Loss Vs Iteration Plot
<img width="497" alt="graph" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/5f82ddc6-b2e3-46b4-950e-3dfdc52eba46">

### Test Data Root Mean Squared Error

<img width="500" alt="image" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/0af15ec1-c109-4091-b1e2-5f67a55da1f4">


### New Sample Data Prediction
<img width="341" alt="image" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/b76f5fa6-46a3-4103-8234-15992fcd768d">

## RESULT:
Therefore We successfully developed a neural network regression model for the given dataset.
