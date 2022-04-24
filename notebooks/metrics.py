import numpy as np
import pandas as pd


def RMSLE(y_true, y_pred, *args):
    rmsle = np.mean((np.log(1 + y_pred) - np.log(1 + y_true)) ** 2) ** 0.5
    return rmsle


def MAPE(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / y_true)


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def MAPE_w(y_true, y_pred):
    return MAPE(np.exp(y_true), np.exp(y_pred), y_true, y_pred)


def MAPE_optim(y_true, y_pred):
    return "MAPE", MAPE(np.exp(y_true), np.exp(y_pred), y_true, y_pred), False
