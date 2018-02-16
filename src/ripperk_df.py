import pandas as pd
import numpy as np

class ripperk(object):
    def __init__(self, prun_ratio, dl_threshold):
        self.prun_ratio = prun_ratio
        self.dl_threshold = dl_threshold

    def fit(self, df):
        pass

    def predict(self, df):
        pass

    def _get_conditions(self, df):
        df.dtypes