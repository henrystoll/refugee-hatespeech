"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    split_stratified_into_train_val_test,
    oversample,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=split_stratified_into_train_val_test,
            #     inputs=["model_input_combined", ["label"]],
            #     outputs=["training_set", "validation_set", "test_set"],
            #     # name="Split Combined Input Dataset into Train Val and Test Set",
            #     name="Split",
            # ),
            node(
                func=oversample,
                inputs="training_set",
                outputs="os_training_set",
                # name="Oversampling the Training Set",
                name="Oversampling",
            )
        ]
    )
