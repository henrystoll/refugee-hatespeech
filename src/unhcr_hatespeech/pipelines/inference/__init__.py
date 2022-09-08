"""
Overview
========

Downloads a Huggingface tokenizer and transformer model from https://huggingface.co/unhcr/hatespeech-detection, 
which are used to calculate the predictions in a Huggingface transformer pipeline.

Pipeline inputs
===============
All the input datasets need to be represented as parquet datasets.

The following datasets are used as input:
| `unhcr_data`, `hatecheck_data`, `test_set` (from the train, val, test split of the combined_dataset)

Pipeline outputs
================
    A Parquet dataset for each of the input datasets with the predictions.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"