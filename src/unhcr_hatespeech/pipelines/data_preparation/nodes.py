import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# TODO: add to config
def split_stratified_into_train_val_test(
    df: pd.DataFrame,
    stratify_cols=["label"],
    frac_train=0.6,
    frac_val=0.15,
    frac_test=0.25,
    random_state=42,
):
    """
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_cols : list[str]
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    """

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(
            "fractions %f, %f, %f do not add up to 1.0"
            % (frac_train, frac_val, frac_test)
        )

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError(
            "fractions %f, %f, %f do not add up to 1.0"
            % (frac_train, frac_val, frac_test)
        )

    X = df  # Contains all columns.
    # Dataframe of just the column on which to stratify.
    y = df[stratify_cols]

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(
        X, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_state
    )

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(
        df_temp,
        y_temp,
        stratify=y_temp,
        test_size=relative_frac_test,
        random_state=random_state,
    )

    assert len(df) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def oversample(
    df: pd.DataFrame,
    # TODO: make input
    oversample_counts: dict = {
        "dynhs": 1,
        "cad": 4,
        "hatemmoji": 5,
        "ghc": 3,
        "hatexplain": 2,
    },
) -> pd.DataFrame:
    """
    Oversamples the hate and offensive classes for the
    specified datasets in the input dictionary.
    """
    for dataset, count in oversample_counts.items():
        for _ in range(count):
            df_new = df[(df["dataset"] == dataset) & (df["label"] > 0)]
            df = pd.concat([df, df_new])
    return shuffle(df, random_state=42)
