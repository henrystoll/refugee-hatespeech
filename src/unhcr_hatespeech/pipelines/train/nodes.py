"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.1
"""
import numpy as np
import pandas as pd

from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


def train(df: pd.DataFrame, model_params, num_labels=3, use_gpu=False):
    """
    Train a model using the given dataset.
    Args:
        df: A pandas DataFrame containing the training data. This will be tokenized.
        model_params: A dictionary of model parameters.
        num_labels: The number of labels to predict.
        use_gpu: Whether to train locally using the GPU or the local CPU.
    """

    tokenized_dataset = load_tokenized_dataset(df)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_params.huggingface, num_labels=num_labels
    )

    metric_string = "f1"
    train_rows = tokenized_dataset.num_rows["train"]
    num_train_epochs = 5
    eval_save_steps = train_rows // 2
    eval_save_steps_adj = (
        eval_save_steps
        // model_params.gradient_accumulation_steps
        // model_params.batch_size
    )
    warmup_steps = (
        train_rows
        * num_train_epochs
        // model_params.gradient_accumulation_steps
        // model_params.batch_size
        // 100
    )

    print(f"{eval_save_steps_adj=}")
    print(f"{warmup_steps=}")

    training_args = TrainingArguments(
        ### training hpo
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=model_params.batch_size,
        per_device_eval_batch_size=model_params.batch_size,
        learning_rate=2e-5,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        gradient_accumulation_steps=model_params.gradient_accumulation_steps,
        ### evaluation
        evaluation_strategy="steps",
        eval_steps=eval_save_steps_adj,
        eval_accumulation_steps=4,
        metric_for_best_model=metric_string,
        ### training optimizations
        fp16=not use_gpu,
        deepspeed=None if use_gpu else "./ds_config.json",
        ### output
        load_best_model_at_end=True,  # model will be saved after each evaluation, only best will be uploaded
        output_dir="./results",
        save_steps=eval_save_steps_adj,
        ### logging
        logging_dir="./logs",
        # report_to=["wandb"] if do_report else "none", TODO: use own reporting solution
        logging_steps=5,
    )

    metric = load_metric(metric_string)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(
            predictions=predictions, references=labels, average="macro"
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
