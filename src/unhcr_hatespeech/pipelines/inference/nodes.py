import pandas as pd
from transformers import AutoModelForSequenceClassification, pipeline, AutoTokenizer

# TODO: add to config
def _download_tokenizer(tokenizer_identifier: str = "unhcr/hatespeech-detection"):
    return AutoTokenizer.from_pretrained(tokenizer_identifier)

# TODO: add to config
def _download_classifier(classifier_identifier: str = "unhcr/hatespeech-detection"):
    id2label = {
        0: "Normal",
        1: "Offensive",
        2: "Hate speech",
    }

    label2id = {id2label[i]: i for i in id2label}
    
    # TODO: add num_labels to config
    return AutoModelForSequenceClassification.from_pretrained(
        classifier_identifier, num_labels=3, id2label=id2label, label2id=label2id
    )


def run_inference(data: pd.DataFrame, text_col: str = "text", local: bool = True) -> pd.DataFrame:
    tokenizer = _download_tokenizer()
    classifier = _download_classifier()

    inference_pipeline = pipeline(
        task="text-classification",
        tokenizer=tokenizer,
        model=classifier,
        # TODO: add to config
        device= -1 if local else 0,
        return_all_scores=True,
        max_length=128,
        padding=True,
        truncation=True,
    )

    preds = inference_pipeline(data[text_col].tolist())
    return pd.DataFrame(map(lambda x: {d["label"]: d["score"] for d in x}, preds))
