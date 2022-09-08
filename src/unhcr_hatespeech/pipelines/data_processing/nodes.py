from functools import partial
import pandas as pd
import emoji

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def clean_text(
    df: pd.DataFrame,
    text_col="text",
    lower=True,
    user_mention=True,
    url=True,
    hashtag=True,
    emoji_to_text=True,
    linebreak=True,
    blank_spaces=True,
    new_line=True,
) -> pd.DataFrame:
    """
    :param df: Pandas DataFrame
    :param text_col: Name of the column containing the text
    :param lower: Convert text to lower case
    :param user_mention: Remove user mentions
    :param url: Remove urls
    :param hashtag: Remove hashtags
    :param emoji_to_text: Convert emojis to text
    :param linebreak: Remove linebreaks
    :param blank_spaces: Remove blank spaces
    :param new_line: Remove new line
    :returns: Pandas DataFrame with cleaned text
    """

    data = df.copy()

    def replace_text_in_df(data: pd.Series, pattern: str, replacement: str):
        data[text_col] = data[text_col].str.replace(pattern, replacement, regex=True)

    replace = partial(replace_text_in_df, data)

    if lower:
        data[text_col] = data[text_col].str.lower()

    if user_mention:
        replace(r'@[A-Za-z0-9._!"#%£$&/{}()=?´*><\|:;]+', "@USER")
        replace("<user>", "@USER")

    if url:
        replace(r"http\S+", "URL")
        replace("@url", "URL")

    if hashtag:
        replace("#", "")

    if emoji_to_text:
        data[text_col] = data[text_col].map(
            lambda row: " ".join([emoji.demojize(word) for word in row.split()])
        )

    if linebreak:
        replace(r"\[linebreak\]", "")

    if blank_spaces:
        replace(r"\s+", " ")

    if new_line:
        replace(r"\n", "")

    return data


def preprocess_cad(raw: pd.DataFrame) -> pd.DataFrame:
    hate_speech_keywords = [
        "IdentityDirectedAbuse / derogation",
        "IdentityDirectedAbuse / threatening language",
        "IdentityDirectedAbuse / dehumanization",
    ]
    offensive_keywords = []
    toxic_keywords = ["IdentityDirectedAbuse / animosity"]

    def _transform_labels(data, keywords: list):
        return [1 if label in keywords else 0 for label in data["annotation_Secondary"]]

    df = raw[["meta_text"]].copy()

    df["hate_speech"] = _transform_labels(raw, hate_speech_keywords)
    df["offensive"] = _transform_labels(raw, offensive_keywords)
    df["toxic"] = _transform_labels(raw, toxic_keywords)

    df = df[df["hate_speech"] == 1]

    # drop na
    df = df.dropna(subset=["meta_text"])
    df = df[
        (df["meta_text"] != "[removed]")
        & (df["meta_text"] != "[deleted]")
        & (~df["meta_text"].str.contains("Your post has been removed"))
    ]

    # rename columns and set dataset name
    df = df.rename(columns={"meta_text": "text"})

    df["dataset"] = "cad"
    return df


def preprocess_civil(raw: pd.DataFrame) -> pd.DataFrame:
    civil_labels = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "sexual_explicit",
        "identity_attack",
        "insult",
        "threat",
    ]

    raw = raw[["comment_text"] + civil_labels]
    df = raw[["comment_text"]].rename(columns={"comment_text": "text"})

    def _apply_threshold(value, threshold=0.5):
        return 1 if value > threshold else 0

    raw = raw[civil_labels].applymap(_apply_threshold)

    df["hate_speech"] = raw["identity_attack"]

    df["toxic"] = raw["toxicity"] + raw["severe_toxicity"]

    df["offensive"] = raw["insult"] + raw["obscene"] + raw["threat"]

    # undersample normal class
    rows_combined = df["offensive"] + df["toxic"] + df["hate_speech"]

    df_negative = df[rows_combined == 0].sample(75_000)
    df_positive = df[rows_combined > 0]

    df = pd.concat([df_positive, df_negative])
    df["dataset"] = "civil"
    return df


def preprocess_davidson(raw: pd.DataFrame) -> pd.DataFrame:
    # transform labels
    davidson_labels = pd.get_dummies(raw["class"]).rename(
        columns={0: "hate_speech", 1: "offensive"}
    )
    davidson_labels = davidson_labels.drop([2], axis=1)

    # transform
    df = raw[["tweet"]]
    df = df.rename(columns={"tweet": "text"})
    df = pd.concat([df, davidson_labels], axis=1)
    df["dataset"] = "davidson"
    return df


def preprocess_dynhs(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["text", "label"]]
    df = df.replace({"label": {"hate": 1, "nothate": 0}})
    df = df.rename(columns={"label": "hate_speech"})
    df["dataset"] = "dynhs"
    return df


def preprocess_ghc(raw: pd.DataFrame) -> pd.DataFrame:
    # transform
    df = raw.drop(["hd", "vo", "cv"], axis=1)

    # add hd and cv and if they are both are one set them one
    df["hate_speech"] = raw["hd"] + raw["cv"]
    df["offensive"] = raw["vo"]
    df["dataset"] = "ghc"
    return df


def preprocess_hatemoji(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["text", "label_gold"]].rename(columns={"label_gold": "hate_speech"})
    df["dataset"] = "hatemoji"
    return df


def preprocess_hateval(raw: pd.DataFrame) -> pd.DataFrame:
    # show that AG is a subset of HS
    raw[(raw["HS"] == 0) & (raw["AG"] == 1)].count()
    df = raw.drop(["id", "TR", "AG"], axis=1)

    df = df.rename(columns={"HS": "hate_speech"})
    df["dataset"] = "hateval"
    return df


def preprocess_hatexplain(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["text", "label"]]
    df = pd.get_dummies(df, columns=["label"])

    df = df.rename(
        columns={"label_hatespeech": "hate_speech", "label_offensive": "offensive"}
    )
    df = df.drop("label_normal", axis=1)
    df["dataset"] = "hatexplain"
    return df


def preprocess_hasoc(raw: pd.DataFrame) -> pd.DataFrame:
    task_col_names = [f"task_{index}" for index in range(1, 4)]
    df = raw.drop(["text_id"] + task_col_names, axis=1)

    df["hate_speech"] = raw["task_2"].map({"HATE": 1, "PRFN": 0, "OFFN": 0, "NONE": 0})
    df["offensive"] = raw["task_2"].map({"HATE": 0, "PRFN": 0, "OFFN": 1, "NONE": 0})
    df["dataset"] = "hasoc"
    return df


def preprocess_slur(raw: pd.DataFrame) -> pd.DataFrame:
    def _transform_labels(data, keyword: str):
        return [1 if label == keyword else 0 for label in data]

    # transform
    df = raw.drop(
        [
            "id",
            "link_id",
            "parent_id",
            "score",
            "subreddit",
            "author",
            "slur",
            "disagreement",
            "gold_label",
        ],
        axis=1,
    )
    df["hate_speech"] = _transform_labels(raw["gold_label"], "DEG")
    df = df.rename(columns={"body": "text"})
    df["dataset"] = "slur"
    return df


def preprocess_ousid(raw: pd.DataFrame) -> pd.DataFrame:
    multi_labels = raw["sentiment"].unique()
    label_collection = list(
        set(lab for label in multi_labels for lab in label.split("_"))
    )

    def extract_labels(data, labels):
        for label in labels:
            data[label] = [
                1 if label in multi_label else 0 for multi_label in data["sentiment"]
            ]
        return data

    raw = extract_labels(raw, label_collection)
    df = raw[["tweet"] + label_collection].copy()
    df["offensive_disrespectful"] = raw["disrespectful"] + raw["offensive"]
    df = df.drop(["normal", "fearful", "offensive", "disrespectful"], axis=1)
    df = df.rename(
        columns={
            "tweet": "text",
            "hateful": "hate_speech",
            "offensive_disrespectful": "offensive",
            "abusive": "toxic",
        }
    )
    df["dataset"] = "ousid"
    return df


def preprocess_wiki(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["text"] = raw["comment_text"].copy()

    df["hate_speech"] = raw["identity_hate"]
    df["offensive"] = raw["insult"] + raw["obscene"] + raw["threat"]
    df["toxic"] = raw["toxic"] + raw["severe_toxic"]

    # undersample normal class
    rows_combined = df["offensive"] + df["toxic"] + df["hate_speech"]

    df_negative = df[rows_combined == 0].sample(75_000)
    df_positive = df[rows_combined > 0]

    df = pd.concat([df_positive, df_negative])
    df["dataset"] = "wiki"
    return df


def _has_binary_labels(df: pd.DataFrame, labels: list) -> bool:
    return all([set(pd.unique(df[labels])) == {0, 1} for label in labels])


def _binarize_labels(combined: pd.DataFrame, labels: list) -> pd.DataFrame:
    """
    set all summed up values to one, because if a value is > 1 -> 1.
    This can be the case if a label has two classes in the original
    dataset that are considered the same class in standardized set
    """
    min_to_one = partial(min, 1)
    combined[labels] = combined[labels].applymap(min_to_one)
    # check if all labels are binary
    # assert _has_binary_labels(combined, labels)
    return combined


def _has_correct_columns(df: pd.DataFrame, labels: list) -> bool:
    correct_columns = set(labels + ["text", "dataset"])
    return set(df.columns) == correct_columns


def combine_training_datasets(*dfs: list[pd.DataFrame]) -> pd.DataFrame:
    labels = ["hate_speech", "offensive", "toxic"]

    for df in dfs:
        # 1. check: all have correct columns before merging
        for label in labels:
            # if label col does not exist set it to zero
            if label not in df.columns:
                df[label] = 0
        if not _has_correct_columns(df, labels):
            raise KeyError(f"{df['dataset']} is missing columns")

    print("all datasets have correct columns")
    combined = pd.concat(dfs)
    print("concatenated all datasets")
    # 2. check test if there are the correct amount of datasets
    # assert len(combined["dataset"].unique()) == 12

    print("dropping NaN...")
    combined = combined.dropna(axis=0)
    print("reseting index...")
    combined = combined.reset_index(drop=True)
    print("binarizing labels...")
    combined = _binarize_labels(combined, labels)

    # create label columns
    combined["label"] = [
        2 if hs > 0 else 1 if off > 0 else 0 if tox > 0 else 0
        for hs, off, tox in zip(
            combined["hate_speech"], combined["offensive"], combined["toxic"]
        )
    ]

    # drop unnecessary columns
    combined = combined.drop(["hate_speech", "offensive", "toxic"], axis=1)
    return combined


def _replace_template_string(
    data: pd.DataFrame, template: str, token: str
) -> pd.DataFrame:
    data = data[data["case_templ"].str.contains(template, regex=False)].copy()
    data["test_case"] = data["case_templ"].str.replace(template, token, regex=False)
    data["target_ident"] = "refugees"
    return data


def preprocess_hatecheck(df: pd.DataFrame) -> pd.DataFrame:
    # create refugee test cases
    tokens = {
        "[IDENTITY_P]": "refugees",
        "[IDENTITY_S]": "refugee",
        "[IDENTITY_A]": "refugee",
        "[IDENTITY_P_space_add]": "r e f u g e e s",
        "[IDENTITY_S_space_add]": "r e f u g e e",
        "[IDENTITY_P_char_del]": "rfugees",
        "[IDENTITY_S_char_del]": "rfugee",
        "[IDENTITY_P_leet]": "r3fugee",
    }

    df_templates = df.drop_duplicates(subset=["templ_id"])

    refugee_dfs = [
        _replace_template_string(df_templates, template, token)
        for template, token in tokens.items()
    ]
    df = pd.concat([df, *refugee_dfs])

    df = df[["test_case", "label_gold", "target_ident", "direction", "functionality"]]
    df = df.rename(columns={"test_case": "text", "label_gold": "label"})

    # Renaming columns
    df = df.rename(columns={"test_case": "text", "label_gold": "label"})

    # Adding dataset column for tracking
    df["dataset"] = "hatecheck"

    # Remapping labels
    df["label"] = df["label"].map({"hateful": 2, "non-hateful": 0})

    # Handling not specific values
    df["direction"] = df["direction"].replace("-", "not specified")
    df["target_ident"] = df["target_ident"].fillna("not specified")
    df["target_ident"] = df["target_ident"].replace("Muslims", "muslims")

    return df


def preprocess_unhcr(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[["Hit Sentence"]].rename(columns={"Hit Sentence": "text"})
    df = df.drop_duplicates()

    df["label"] = 2
    df["dataset"] = "unhcr"

    return df


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
