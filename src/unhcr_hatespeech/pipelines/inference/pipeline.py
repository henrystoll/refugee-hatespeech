"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_inference


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_inference,
                inputs="test_unhcr",
                outputs="unhcr_predictions",
                name="run_inference_unhcr",
            ),
           # node(
            #    func=run_inference,
             #   inputs="test_set",
              #  outputs="test_set_predictions",
               # name="run_inference_test_set",
            #),
            #node(
             #   func=run_inference,
              #  inputs="test_hatecheck",
               # outputs="hatecheck_predictions",
                #name="run_inference_hatecheck",
            #),
        ]
    )
