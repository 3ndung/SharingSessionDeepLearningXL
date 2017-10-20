import pandas as pd
from keras.models import Sequential
from keras.layers import  *
from keras.utils import plot_model

Data_Belajar = pd.read_csv('Finale_Training.csv')
print Data_Belajar.head(1)

X = Data_Belajar.drop('LEVEL_KEMACETAN', axis=1).values
Y = Data_Belajar[['LEVEL_KEMACETAN']].values

Data_uji = pd.read_csv('Finale_Test.csv')

X_uji = Data_uji.drop(['LEVEL_KEMACETAN'], axis=1).values
Y_uji = Data_uji[['LEVEL_KEMACETAN']].values


model = Sequential()
model.add(Dense(50, input_dim=4, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(
    X,
    Y,
    epochs=100,
    shuffle=True,
    verbose=2
)

Test_Erroe_Rate = model.evaluate(
    X_uji,
    Y_uji,
    verbose=0
)

#plot_model(model, to_file='Model.png')

print("ini Adalah nilai Test Error Rate nya {}".format(Test_Erroe_Rate))
print(Test_Erroe_Rate)

model.save('iNew_Model_Learn.h5')
print("Model Saved ")