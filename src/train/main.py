 import numpy as np

import pandas as pd
from car_price import *

# load data
car_price_df = pd.read_csv('./Dataset/CarPrice_Assignment.csv')
# split dataframe into inputs and targets
features_column = ['carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower', 'citympg', 'highwaympg']
X = car_price_df[features_column].to_numpy()
Y = np.expand_dims(car_price_df['price'].to_numpy(), 1)

# parameters
batch_size = 16
seed = 106
epochs = 3000
display_freq = epochs//50
lr = 1e-2  # learning rate
model_path = f'./Model/model_e3000.pth'  # to save and load models
# initialize app
car_prices_app = CarPriceApp((X, Y), batch_size=batch_size, seed=seed, model_path= model_path)
# uncomment to load pre-trained model
# car_prices_app.load_model(model_path)
# uncomment to train model
# car_prices_app.train_model(epochs=epochs, display_freq=display_freq, save_model=True)       return x, y
