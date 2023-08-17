# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

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

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
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
X = dataset[['Input']].values
Y = dataset[['Output']].values
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state = 20)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train_scale = Scaler.transform(x_train)
my_brain = Sequential([
    Dense(units = 4, activation = 'relu' , input_shape=[1]),
    Dense(units = 6),
    Dense(units = 1)

])
my_brain.compile(optimizer='rmsprop',loss='mse')
my_brain.fit(x=x_train_scale,y=y_train,epochs=20000)
loss_df = pd.DataFrame(my_brain.history.history)
loss_df.plot()
x_test1 = Scaler.transform(x_test)
my_brain.evaluate(x_test1,y_test)
X_n1 = [[30]]
input_scaled = Scaler.transform(X_n1)
my_brain.predict(input_scaled)
```
## Dataset Information

<img width="196" alt="set" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/5d715246-ca1b-42df-830a-80697df8cf1b">


## OUTPUT:

### Training Loss Vs Iteration Plot
<img width="497" alt="graph" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/5f82ddc6-b2e3-46b4-950e-3dfdc52eba46">

### Test Data Root Mean Squared Error

<img width="568" alt="1" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/f598854f-57e3-4186-8379-bd8140ac5d68">


### New Sample Data Prediction

<img width="454" alt="2" src="https://github.com/Dhanireddy-Amarnthreddy/basic-nn-model/assets/94165103/ff966abd-8418-443b-93e2-b4dd09d219ff">

## RESULT:
Therefore We successfully developed a neural network regression model for the given dataset.
