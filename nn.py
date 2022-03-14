from argparse import ArgumentParser
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import math
from sklearn import metrics

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--real_train", type=str, default='isic_usecase.csv')
    parser.add_argument("--synth_train", type=str, default='isic_ctgan.csv',
                        help='path to load generated data')
    parser.add_argument("--real_test", type=str, default='isic_predict.csv')
    parser.add_argument("--model_save", type=str, default='isic_ctgan.pkl',
                        help='path to save trained model')
    parser.add_argument("--model", type=str, default='ctgan')
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--num_samples", type=int, default=100000)
    args = parser.parse_args()

    ## neural network
    model = Sequential()
    # Adding the input layer and first hidden layer
    model.add(Dense(100, activation = 'relu', input_dim = 19))
    # Adding the second hidden layer
    model.add(Dense(units = 50, activation = 'relu'))
    # Adding other hidden layers
    model.add(Dense(units = 50, activation = 'relu'))
    model.add(Dense(units = 50, activation = 'relu'))
    #Finally, adding the output layer
    model.add(Dense(units = 1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mae','mse'])
    model.summary()

    ## data
    # Loading data into dataframes
    los_use = pd.read_csv(args.real_train)
    los_predict = pd.read_csv(args.real_test)
    los_ctgan = pd.read_csv(args.synth_train)

    #Selecting the required columns - features and target variable.
    X_test = los_predict.iloc[:,5:] # .join(los_predict.iloc[:,:4])
    y_test = los_predict.iloc[:,4]
    # real
    y_train = los_use.iloc[:,4]
    X_train = los_use.iloc[:,5:]#.join(los_predict.iloc[:,:4]) # shape 19
    # synthetic
    X_train_ctgan = los_ctgan.iloc[:,5:]
    y_train_ctgan = los_ctgan.iloc[:,4]

    # Scaling on predictor variables of test and train
    # is done to normalize the dataset. This step is important for regularization
    scaler = StandardScaler()
    scaler.fit(X_train)
    x_train_scaled=scaler.transform(X_train)
    x_test_scaled=scaler.transform(X_test)

    scaler = StandardScaler()
    scaler.fit(X_train_ctgan)
    x_train_ctgan_scaled=scaler.transform(X_train_ctgan)

    neur_net_model = model.fit(x_train_scaled, y_train, batch_size=50, 
                               epochs = 100, verbose = 0, validation_split = 0.2)

    neur_net_predict = model.predict(x_test_scaled)
    rms_nn = math.sqrt(metrics.mean_squared_error(y_test, neur_net_predict))
    print('RMSE = {}'.format(rms_nn))
    ms_nn = metrics.mean_squared_error(y_test, neur_net_predict)
    print('MSE = {}'.format(ms_nn))
    mae_nn = metrics.mean_absolute_error(y_test, neur_net_predict)
    print('MAE = {}'.format(mae_nn))
    print('Explained Variance Score:', metrics.explained_variance_score(y_test, neur_net_predict))
    print('Coefficient of Determination:', metrics.r2_score(y_test, neur_net_predict))

    neur_net_model = model.fit(x_train_ctgan_scaled, y_train_ctgan, batch_size=50,
                               epochs = 100, verbose = 0, validation_split = 0.2)
    neur_net_predict_ctgan = model.predict(x_test_scaled)

    rms_nn = math.sqrt(metrics.mean_squared_error(y_test, neur_net_predict_ctgan))
    print('RMSE = {}'.format(rms_nn))
    ms_nn = metrics.mean_squared_error(y_test, neur_net_predict_ctgan)
    print('MSE = {}'.format(ms_nn))
    mae_nn = metrics.mean_absolute_error(y_test, neur_net_predict_ctgan)
    print('MAE = {}'.format(mae_nn))
    print('Explained Variance Score:', metrics.explained_variance_score(y_test, neur_net_predict_ctgan))
    print('Coefficient of Determination:', metrics.r2_score(y_test, neur_net_predict_ctgan))
