import pandas as pd


def load_expression(path: str) -> pd.DataFrame:
    """Load RNA-seq expression data"""
    return pd.read_csv(path, low_memory=False)


def load_ic50(path: str) -> pd.DataFrame:
    """Load drug response data"""
    return pd.read_csv(path)


def load_mapping(path: str) -> pd.DataFrame:
    """Load model mapping data"""
    return pd.read_csv(path)