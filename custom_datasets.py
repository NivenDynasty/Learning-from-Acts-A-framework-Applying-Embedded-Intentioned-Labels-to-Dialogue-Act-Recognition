from torchtext.data import Dataset, Example

import pandas as pd


# Dataset for pandas.DataFrame.
class DataFrameDataset(Dataset):

    def __init__(self, df: pd.DataFrame, fields: list):
        super(DataFrameDataset, self).__init__(
            [Example.fromlist(list(r), fields) for _, r in df.iterrows()], fields
        )