"""
## Overview
Raw
--------
Preprocess each dataset:
For the (academic) datasets used as training data: map the label to normal, toxic, offensive or hate speech.
For all datasets: preprocess the text (remove URLs, emojis, etc.)

Primary
--------
Combine datasets into one training dataset. 
Standardize into one `label` colummn: 0 = normal, 1 = toxic, 2 = hate speech



## Pipeline inputs
12 academic datasets:
- cad
- civil
- davidson
- dynhs
- ghc
- hasoc
- hatemoji
- hateval
- hatexplain
- ousid
- slur
- wikipedia
## Pipeline outputs
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]

__version__ = "0.1"
