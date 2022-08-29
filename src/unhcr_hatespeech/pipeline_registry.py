"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from unhcr_hatespeech.pipelines import data_processing as dp, data_preparation as data_prep


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_processing_pipeline = dp.create_pipeline()
    data_preparation_pipeline = data_prep.create_pipeline()

    return {
        "__default__": pipeline([data_processing_pipeline, data_preparation_pipeline]),
        # "__default__": data_processing_pipeline,
        "dp": data_processing_pipeline,
        "data_prep": data_preparation_pipeline,
    }
