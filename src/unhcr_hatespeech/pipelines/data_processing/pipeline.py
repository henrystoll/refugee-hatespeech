"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    combine_and_clean_input,
    preprocess_cad,
    preprocess_civil,
    preprocess_davidson,
    preprocess_dynhs,
    preprocess_ghc,
    preprocess_hasoc,
    preprocess_hatemoji,
    preprocess_hateval,
    preprocess_hatexplain,
    preprocess_slur,
    preprocess_ousid,
    preprocess_wiki,
    preprocess_hatecheck,
    preprocess_unhcr,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_cad,
                inputs="train_cad",
                outputs="preprocessed_cad",
                name="preprocess_cad_node",
            ),
            #node(
             #   func=preprocess_civil,
              #  inputs="train_civil",
               # outputs="preprocessed_civil",
                #name="preprocess_civil_node",
            #),
            node(
                func=preprocess_davidson,
                inputs="train_davidson",
                outputs="preprocessed_davidson",
                name="preprocess_davidson_node",
            ),
            node(
                func=preprocess_dynhs,
                inputs="train_dynhs",
                outputs="preprocessed_dynhs",
                name="preprocess_dynhs_node",
            ),
            node(
                func=preprocess_ghc,
                inputs="train_ghc",
                outputs="preprocessed_ghc",
                name="preprocess_ghc_node",
            ),
            node(
                func=preprocess_hasoc,
                inputs="train_hasoc",
                outputs="preprocessed_hasoc",
                name="preprocess_hasoc_node",
            ),
            node(
                func=preprocess_hatemoji,
                inputs="train_hatemoji",
                outputs="preprocessed_hatemoji",
                name="preprocess_hatemoji_node",
            ),
            node(
                func=preprocess_hateval,
                inputs="train_hateval",
                outputs="preprocessed_hateval",
                name="preprocess_hateval_node",
            ),
            node(
                func=preprocess_hatexplain,
                inputs="train_hatexplain",
                outputs="preprocessed_hatexplain",
                name="preprocess_hatexplain_node",
            ),
            node(
                func=preprocess_ousid,
                inputs="train_ousid",
                outputs="preprocessed_ousid",
                name="preprocess_ousid_node",
            ),
            node(
                func=preprocess_slur,
                inputs="train_slur",
                outputs="preprocessed_slur",
                name="preprocess_slur_node",
            ),
            node(
                func=preprocess_wiki,
                inputs="train_wiki",
                outputs="preprocessed_wiki",
                name="preprocess_wiki_node",
            ),
            node(
                func=combine_and_clean_input,
                inputs=[
                    "preprocessed_cad",
                    #"preprocessed_civil",
                    "preprocessed_davidson",
                    "preprocessed_dynhs",
                    "preprocessed_ghc",
                    "preprocessed_hasoc",
                    "preprocessed_hatexplain",
                    "preprocessed_hatemoji",
                    "preprocessed_hateval",
                    "preprocessed_ousid",
                    "preprocessed_slur",
                    "preprocessed_wiki",
                ],
                outputs="model_input_combined",
            ),
            node(
                func=preprocess_hatecheck,
                inputs="raw_hatecheck",
                outputs="test_hatecheck",
                name="preprocess_hatecheck_node",
            ),
            node(
                func=preprocess_unhcr,
                inputs="raw_unhcr",
                outputs="test_unhcr",
                name="preprocess_unhcr_node",
            ),
        ]
    )
