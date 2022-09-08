import pandas as pd
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer

# TODO: add to config
def _download_tokenizer(tokenizer_identifier: str = "unhcr/hatespeech-detection"):
    """
    Downloads the tokenizer that is associated 
    with the model from https://huggingface.co/unhcr/hatespeech-detection

    Parameters
    ----------
    tokenizer_identifier : str
        The name of Huggingface project that the tokenizer is associated with

    Returns
    -------
    tokenizer
        Huggingface transformer tokenizer
    """
    return AutoTokenizer.from_pretrained(tokenizer_identifier)

# TODO: add to config
def _download_classifier(classifier_identifier: str = "unhcr/hatespeech-detection"):
    """
    Downloads the model from https://huggingface.co/unhcr/hatespeech-detection

    Parameters
    ----------
    classifier_identifier : str
        The name of Huggingface project that the model is associated with

    Returns
    -------
    model
        Huggingface transformer model

    Dict
        Label mapping from ids to labels

    """
    id2label = {
        0: "Normal",
        1: "Offensive",
        2: "Hate speech",
    }

    label2id = {id2label[i]: i for i in id2label}

    # TODO: add num_labels to config
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier_identifier, num_labels=3, id2label=id2label, label2id=label2id
    )
    
    return model, id2label


def run_inference(data: pd.DataFrame, text_col: str = "text", local: bool = True) -> pd.DataFrame:
    """
    Creates a transformer pipeline and performs inference

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame that contains a column with text samples
    
    text_col : str
        The column name of the text column
    
    local : bool
        Indicates whether the inference should be run locally or not

    Returns
    -------
    Pandas DataFrame
    """
    tokenizer = _download_tokenizer()
    classifier, id2label = _download_classifier()

    inference_pipeline = pipeline(
        task="text-classification",
        tokenizer=tokenizer,
        model=classifier,
        # TODO: add to config
        device= -1 if local else 0,
        top_k=3,
        max_length=128,
        padding=True,
        truncation=True,
    )

    preds = inference_pipeline(data[text_col].tolist())
    preds_df = pd.DataFrame(map(lambda x: {d["label"]: d["score"] for d in x}, preds))

    output_df = data.copy()
    output_df["Normal"] = preds_df["Normal"].tolist()
    output_df["Hate speech"] = preds_df["Hate speech"].tolist()
    output_df["Offensive"] = preds_df["Offensive"].tolist()
    output_df["Predicted_Label"] = output_df[id2label.values()].idxmax(axis=1).rename("pred_label").tolist()
   
    return output_df
