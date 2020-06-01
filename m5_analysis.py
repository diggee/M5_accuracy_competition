# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:37:06 2020

@author: diggee
"""

#%% imports

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

#%% reading data

def read_data():
        
    calender = pd.read_csv('calendar.csv', index_col = 'date', parse_dates = True)
    calender.index.freq = 'D'
    
    price = pd.read_csv('sell_prices.csv')
    sales = pd.read_csv('sales_train_validation.csv')
    submission = pd.read_csv('sample_submission.csv')
    
    return price, sales, submission

#%% memory reduction

def reduce_mem_usage(df, verbose = True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

#%% data preprocessing
    
def preprocessing(sales):

    sales.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis = 1, inplace = True)
    sales = sales.T
    sales.columns = sales.iloc[0,]
    sales.drop(index = 'id', inplace = True)    
    return sales

#%% train test split
    
def data_split(sales, n_obs):
    train = sales[-n_obs*4:-n_obs]
    validation = sales[-n_obs:]
    return train, validation

#%% data scaling

def data_scaler(train, test, n_input):
    
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train)
    scaled_validation = scaler.transform(validation)
    return scaled_validation, scaled_train, scaler

#%% data series generator
    
def series_generator(scaled_train, scaled_validation, n_input):
    
    train_generator = TimeseriesGenerator(scaled_train, scaled_train, length = n_input, batch_size = 1)
    validation_generator = TimeseriesGenerator(scaled_validation, scaled_validation, length = n_input, batch_size = 1)
    return train_generator, validation_generator

#%% NN model
    
def neural_network(n_units, n_input, n_features, train_generator, validation_generator):    

    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(int(n_units), return_sequences = True), input_shape = (n_input, n_features)))
    model.add(Bidirectional(CuDNNLSTM(int(n_units/2))))
    # model.add(Dense(int(n_features/256), 'relu'))
    # model.add(Dense(int(n_features/128), 'relu'))    
    model.add(Dense(n_features))
    
    early_stop = EarlyStopping(monitor = 'val_loss', patience = 20)
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 20, min_lr = 0.001)
    checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', save_best_only = True, mode = 'min')
    
    model.compile('adam', loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
    model.summary()
    history = model.fit(train_generator, epochs = 500, verbose = 1, 
                        validation_data = validation_generator, 
                        callbacks = [checkpoint, reduce_lr, early_stop])
    return model, history

#%% make plots
    
def make_plots(history):
    plt.figure()
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.legend()
    
    plt.figure()
    plt.plot(history.history['root_mean_squared_error'], label = 'RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label = 'val_RMSE')
    plt.legend()
    
#%% predicton of sales
    
def sales_prediction(model, n_input, n_features, scaled_train, scaler):

    predictions = []
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    
    for i in range(len(validation)):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis = 1)
        
    predictions = scaler.inverse_transform(predictions)
    return predictions

#%% prediction plots

def make_prediction_plots(validation, predictions, random_feature):
    plt.figure()
    plt.title(f'feature column {random_feature}')
    plt.plot(validation.iloc[:, random_feature], label = 'test data')
    plt.plot(predictions[:, random_feature], label = 'prediction')
    plt.legend()        

#%% main

if __name__ == '__main__':
    
    n_obs = 28
    n_input = 7
    n_units = 128
    random_feature = np.random.randint(0, 30490)
    
    price, sales, submission = read_data()
    print(f'price data memory usage = ~{price.memory_usage().sum()/1e6} MB')
    print(f'sales data memory usage = ~{sales.memory_usage().sum()/1e6} MB')
    price = reduce_mem_usage(price)
    sales = reduce_mem_usage(sales)
    sales = preprocessing(sales)
    train, validation = data_split(sales, n_obs)    
    scaled_validation, scaled_train, scaler = data_scaler(train, validation, n_input)
    train_generator, validation_generator = series_generator(scaled_train, scaled_validation, n_input)  
    model, history = neural_network(n_units, n_input, train.shape[1], train_generator, validation_generator)
    make_plots(history)
    
    predictions = sales_prediction(model, n_input, train.shape[1], scaled_train, scaler)
    make_prediction_plots(validation, predictions, random_feature)
    submission.iloc[:30490,1:] = predictions.T
    submission.to_csv('sample_submission.csv', index = False)    