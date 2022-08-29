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
            node(
                func=split_stratified_into_train_val_test,
                inputs=["model_input_combined", "label"],
                outputs=["training_set", "validation_set", "test_set"],
                name="split_combined_input_dataset_into_train_val_and_test_set",
            ),
            node(
                func=oversample,
                inputs="training_set",
                outputs="os_training_set",
                name="oversampling_the_training_set",
            )
        ]
    )
