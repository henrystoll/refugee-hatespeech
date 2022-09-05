"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_interference
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_interference,
                inputs="test_unhcr",
                outputs="tt",
                name="run_interference",
            )
        ]
    )
