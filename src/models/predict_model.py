import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib


def load_models(sj_path, iq_path):
    sj_model = joblib.load(sj_path)
    iq_model = joblib.load(iq_path)
    return sj_model, iq_model

def generate_predictions(data, sjmodel, iqmodel):
    sj = data[data.sj==1]
    iq = data[data.sj==0]

    sj_pred = sjmodel.predict(sj).tolist()
    iq_pred = iqmodel.predict(iq).tolist()

    return sj_pred + iq_pred
