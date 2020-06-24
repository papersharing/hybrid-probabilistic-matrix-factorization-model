import pandas as pd

from ..utils.data_utils import get_zip_file


def load_data():
    """Loads MovieLens 1M dataset.

    Returns:
        Tuple of numpy array (x, y)
    """

    #URL = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    #FILE_PATH = 'ml-1m/ratings.dat'

    #file = get_zip_file(URL, FILE_PATH)
    df = pd.read_csv('/media/a709/Document/zhong/ConvMF/data/movielens/ml_1m/ml-1m_ratings.dat', sep='::', header=None, engine='python')

    return df.iloc[:, :2].values, df.iloc[:, 2].values
