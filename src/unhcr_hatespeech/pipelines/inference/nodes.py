import pandas as pd
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer


def download_tokenizer(tokenizer_identifier: str = "unhcr/hatespeech-detection"):
    return AutoTokenizer.from_pretrained(tokenizer_identifier)


def download_classifier(classifier_identifier: str = "unhcr/hatespeech-detection"):
    id2label = {
        0: "Normal",
        1: "Offensive",
        2: "Hate speech",
    }

    label2id = {id2label[i]: i for i in id2label}

    return AutoModelForSequenceClassification.from_pretrained(
        classifier_identifier, num_labels=3, id2label=id2label, label2id=label2id
    )


def run_inference(data: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    tokenizer = download_classifier()
    classifier = download_classifier()

    inference_pipeline = pipeline(
        task="text-classification",
        tokenizer=tokenizer,
        model=classifier,
        # device=0,
        return_all_scores=True,
        max_length=128,
        padding=True,
        truncation=True,
    )

    preds = inference_pipeline(data[text_col].tolist())
    return pd.DataFrame(map(lambda x: {d["label"]: d["score"] for d in x}, preds))
