from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
import pandas as pd
import numpy as np

TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2021-10-01'
TEST_START_DATE = '2021-10-01'
TEST_END_DATE = '2023-03-01'

df = YahooDownloader(start_date=TRAIN_START_DATE,
                     end_date=TEST_END_DATE,
                     ticker_list=DOW_30_TICKER).fetch_data()

# INDICATORS = ['rsi_14', 'macd', 'boll', 'atr_14', 'volume', 'adx_14', 'kdjk']
INDICATORS = [
    "rsi_14",
    "macd",
    "boll",
    "atr_14",
    "volume",
    "adx",
    "kdjk",
]
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list=INDICATORS,
                     use_turbulence=True,
                     user_defined_feature=False)

processed = fe.preprocess_data(df)
processed = processed.copy()
processed = processed.fillna(0)
processed = processed.replace(np.inf, 0)

print(processed.head())
