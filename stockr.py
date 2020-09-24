import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from yahoo_fin import stock_info as si
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
import argparse
from datetime import datetime
from progress.bar import PixelBar
from colorama import Fore

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Window size or the sequence length
N_STEPS = 70  # 150
# Lookup step, 1 is the next day
LOOKUP_STEP = 1  # 7
TICKER = "ACB"  # "PYPL"

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

bar = PixelBar()

date_string = datetime.now().strftime("%Y-%m-%d")

PORTFOLIO = ["ACB", "AVID", "HMMJ.TO", "BTB-UN.TO", "NWH-UN.TO", "OGI", "PYPL"]


def progressbar(epilog, current, max):
    global bar
    if current == 0:
        bar = PixelBar()
        width, height = os.get_terminal_size()
        width -= len(f"{max}/{max}")
        width -= len(epilog)
        width -= 2  # ears
        width -= 3  # number of spaces
        bar = PixelBar(epilog, max=max, width=width)

    # width -= len(f"{max}/{max}")

    bar.next()

    if current == max:
        bar.finish()


class PrintLogs(tf.keras.callbacks.Callback):
    def __init__(self, epochs, ticker):
        self.epochs = epochs
        self.ticker = ticker

    def set_params(self, params):
        params["epochs"] = 0

    def on_epoch_begin(self, epoch, logs=None):
        progressbar(f"{self.ticker}:", epoch, self.epochs)
        # if epoch == self.epochs:
        #    #print(f"Epoch {epoch + 1}/{self.epochs}", end="\n")
        #    progressbar(f"Ticker:", epoch, self.epochs)

        # else:
        #    print(f"Epoch {epoch + 1}/{self.epochs}", end="\r")
        # sys.stdout.write(f"\rEpoch {epoch + 1}/{self.epochs}")
        # sys.stdout.flush()
        # pass

    def on_epoch_end(self, epoch, logs=None):
        # print()
        pass


def main():
    args = parse_arguments()
    tickets = []
    if args.portfolio:
        tickets = PORTFOLIO
        # print(tickets)
    elif args.tickets != "":
        tickets = args.tickets.split(",")
        # print(tickets)
    else:
        print("Please include -p or -t")
        exit()
    results = {}
    for ticker in tickets:
        #  print(f"Analyzing: [ {ticker} ]")
        # print(ticker)
        current_price = round(si.get_data(ticker)["open"].iloc[-1], 2)
        mean_absolute_error, future_price, accuracy = analyze(ticker)
        results[ticker] = {
            "current_price": current_price,
            "mean_absolute_error": mean_absolute_error,
            "future_price": future_price,
            "accuracy": accuracy,
        }
        data = si.get_data(ticker)["open"].iloc[-1]

    # print(results)
    for ticker in results.keys():
        if args.verbose:
            print(f"==== [ {ticker} ] ====")
            print(f"Price       : {results[ticker]['current_price']}")
            print(f"Future Price: {results[ticker]['future_price']}")
            print(f"Mean Abs Err: {results[ticker]['mean_absolute_error']}")
            print(f"Accuracy    : {results[ticker]['accuracy']}")
            print()
        if (
            results[ticker]["future_price"] - results[ticker]["current_price"]
            > results[ticker]["future_price"] * 0.05
        ):
            action = f"== {Fore.GREEN}BUY {ticker}{Fore.RESET}"

        elif (
            results[ticker]["future_price"] - results[ticker]["current_price"]
            < -results[ticker]["future_price"] * 0.05
        ):
            action = f"{Fore.RED}SELL{Fore.RESET}"
        else:
            action = "HOLD"
        print(f"ACTION      : {action}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("-t", "--tickets", help="an integer for the accumulator")
    parser.add_argument(
        "-p", "--portfolio", help="Analyze saved stocks", action="store_true"
    )
    parser.add_argument('-v', "--verbose", help="" action="store_true")
    args = parser.parse_args()
    return args


def analyze(ticker):
    # Window size or the sequence length
    # N_STEPS = 70
    # Lookup step, 1 is the next day
    # LOOKUP_STEP = 1
    # test ratio size, 0.2 is 20%
    TEST_SIZE = 0.2
    # features to use
    FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
    # date now
    date_now = time.strftime("%Y-%m-%d")
    ### model parameters
    N_LAYERS = 3
    # LSTM cell
    CELL = LSTM
    # 256 LSTM neurons
    UNITS = 256
    # 40% dropout
    DROPOUT = 0.4
    # whether to use bidirectional RNNs
    BIDIRECTIONAL = False
    ### training parameters
    # mean absolute error loss
    # LOSS = "mae"
    # huber loss
    LOSS = "huber_loss"
    OPTIMIZER = "adam"
    BATCH_SIZE = 64
    EPOCHS = 400
    # Tesla stock market
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"

    # create these folders if they does not exist
    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("graphs"):
        os.mkdir("graphs")
    if not os.path.isdir(f"graphs/{date_string}"):
        os.mkdir(f"graphs/{date_string}")

    # load the data
    data = load_data(
        ticker,
        N_STEPS,
        lookup_step=LOOKUP_STEP,
        test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS,
    )
    # save the dataframe
    data["df"].to_csv(ticker_data_filename)
    # construct the model
    model = create_model(
        N_STEPS,
        loss=LOSS,
        units=UNITS,
        cell=CELL,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        optimizer=OPTIMIZER,
        bidirectional=BIDIRECTIONAL,
    )
    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(
        os.path.join("results", model_name + ".h5"),
        save_weights_only=True,
        save_best_only=True,
        verbose=5,
    )
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    history = model.fit(
        data["X_train"],
        data["y_train"],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(data["X_test"], data["y_test"]),
        callbacks=[PrintLogs(EPOCHS, ticker)],
        verbose=5,
    )
    model.save(os.path.join("results", model_name) + ".h5")

    data = load_data(
        ticker,
        N_STEPS,
        lookup_step=LOOKUP_STEP,
        test_size=TEST_SIZE,
        feature_columns=FEATURE_COLUMNS,
        shuffle=False,
    )
    # construct the model
    model = create_model(
        N_STEPS,
        loss=LOSS,
        units=UNITS,
        cell=CELL,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        optimizer=OPTIMIZER,
        bidirectional=BIDIRECTIONAL,
    )
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)

    # evaluate the model
    mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=5)
    # calculate the mean absolute error (inverse scaling)
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[
        0
    ][0]
    # print("Mean Absolute Error:", mean_absolute_error)

    # predict the future price
    future_price = predict(model, data)
    # print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")

    plot_graph(model, data, ticker)
    accuracy = get_accuracy(model, data)

    # print(str(LOOKUP_STEP) + ":", "Accuracy Score:", accuracy)

    return mean_absolute_error, future_price, accuracy


def load_data(
    ticker,
    n_steps=50,
    scale=True,
    shuffle=True,
    lookup_step=1,
    test_size=0.2,
    feature_columns=["adjclose", "volume", "open", "high", "low"],
):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result["df"] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df["future"] = df["adjclose"].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df["future"].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(last_sequence)
    # add to result
    result["last_sequence"] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    # split the dataset
    (
        result["X_train"],
        result["X_test"],
        result["y_train"],
        result["y_test"],
    ) = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # return the result
    return result


def create_model(
    sequence_length,
    units=256,
    cell=LSTM,
    n_layers=2,
    dropout=0.3,
    loss="mean_absolute_error",
    optimizer="rmsprop",
    bidirectional=False,
):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(
                    Bidirectional(
                        cell(units, return_sequences=True),
                        input_shape=(None, sequence_length),
                    )
                )
            else:
                model.add(
                    cell(
                        units,
                        return_sequences=True,
                        input_shape=(None, sequence_length),
                    )
                )
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


def predict(model, data):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape(
        (last_sequence.shape[1], last_sequence.shape[0])
    )
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_price


def plot_graph(model, data, ticker):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(
        data["column_scaler"]["adjclose"].inverse_transform(
            np.expand_dims(y_test, axis=0)
        )
    )
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    # last 200 days, feel free to edit that
    plt.plot(y_test[-200:], c="b")
    plt.plot(y_pred[-200:], c="r")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    # plt.show()
    plt.savefig(f"graphs/{date_string}/{ticker}.jpg")
    plt.clf()  # clears the plot for the next ticker


def get_accuracy(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(
        data["column_scaler"]["adjclose"].inverse_transform(
            np.expand_dims(y_test, axis=0)
        )
    )
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(
        map(
            lambda current, future: int(float(future) > float(current)),
            y_test[:-LOOKUP_STEP],
            y_pred[LOOKUP_STEP:],
        )
    )
    y_test = list(
        map(
            lambda current, future: int(float(future) > float(current)),
            y_test[:-LOOKUP_STEP],
            y_test[LOOKUP_STEP:],
        )
    )
    return accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    main()