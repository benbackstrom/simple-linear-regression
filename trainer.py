
from pandas import DataFrame
from pandas import Series
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.metrics import RootMeanSquaredError


def train_model(feature: Series, label: Series, learning_rate, number_epochs, batch_size):
    # Create model
    model = Sequential()
    model.add(Dense(units=1, activation='relu', input_shape=(1,)))
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss='mean_squared_error',
                  metrics=[RootMeanSquaredError()])

    # Train model
    model_hist = model.fit(x=feature, y=label, epochs=number_epochs, batch_size=batch_size)

    # Get root mean squared error and epoch data
    df_hist = DataFrame(model_hist.history)
    root_mean_squared_error = df_hist['root_mean_squared_error']

    # Get weight, bias
    weights = model.get_weights()
    weight = weights[0]
    bias = weights[1]

    return weight, bias, root_mean_squared_error, model_hist.epoch
