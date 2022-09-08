"""
Overview
========

Preprocessing pipeline for the mapping, combining and cleaning of the different datasets.

Raw
---
Preprocess each dataset:
    For the (academic) datasets used as training data: map the label to normal, toxic, offensive or hate speech.

Primary
--------
Training Datasets:
    Combine datasets into one training dataset.
    Standardize labels into one `label` colummn:: 
        0 = normal, 
        1 = toxic, 
        2 = hate speech

All Datasets:
    Clean Text: remove URLs, emojis, special characters, etc.

Model Input
-----------
    Split into train, val, test sets.
    Oversample training set.


Pipeline inputs
===============

12 academic datasets:

#. cad
#. civil
#. davidson
#. dynhs
#. ghc
#. hasoc
#. hatemoji
#. hateval
#. hatexplain
#. ousid
#. slur
#. wikipedia

Test Datasets:

#. hatecheck
#. unhcr

Pipeline outputs
================

Training Datasets:

#. train_set
#. train_set_oversampled

Test Datasets:

#. test_set (for model evaluation and calcualting test metrics like f1 score)
#. test_hatecheck
#. test_unhcr

"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
