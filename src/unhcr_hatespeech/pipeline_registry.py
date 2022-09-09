"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from unhcr_hatespeech.pipelines import (
    data_processing as dp,
    inference,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_processing_pipeline = dp.create_pipeline()
    inference_pipeline = inference.create_pipeline()

    return {
        "__default__": pipeline([data_processing_pipeline, inference_pipeline]),
        # "__default__": data_processing_pipeline,
        "dp": data_processing_pipeline,
        "inference": inference_pipeline,
    }
