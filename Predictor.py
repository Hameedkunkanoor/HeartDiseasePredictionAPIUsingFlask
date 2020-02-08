import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras.layers import Dense,Dropout


model = load_model("HeartDiseasePredictionModel.h5")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
input=np.array([[54,1,1,192,283,0,0,195,0,0.0,2,1,3]])
prediction=model.predict(input)
print(prediction)