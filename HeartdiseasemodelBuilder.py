import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
dataset = pd.read_csv("heart.csv")
from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]
print("The parameters in the dataset are ......")
for i in range(len(dataset.columns)):
 print(dataset.columns[i])
print("The total rows of data present in dataset is "+str(dataset.shape[0]))
Heartdiseaseratio=dataset.target.value_counts()
print("The number of patients without heart disease are"+str(Heartdiseaseratio[0]))
print("The number of patients with heart disease are"+str(Heartdiseaseratio[1]))

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))

model.add(Dense(13,activation='relu'))

model.add(Dense(13,activation='relu'))
model.add(Dense(13,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=1000)
model.save("HeartDiseasePredictionModel.h5")
Y_pred_nn = model.predict(X_test)
#print("The accuracy score achieved using Neural Network is: "+Y_pred_nn.values+" %")
