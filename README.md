# UNHCR Hatespeech Detection
**Creators:** Frederik Gaasdal Jensen & Henry Alexander Stoll

## About this Repo
This repository is built using Kedro, which makes it possible to have a more clear structure of a project. More specifically, Kedro uses nodes and pipelines to orchestrate data flows.

***Node:*** a Python function

***Pipeline:*** a sequence of nodes

To understand Kedro better, take a look at [Kedro documentation](https://kedro.readthedocs.io).

This project provides **two pipelines**:

1. Data Processing
2. Model Inference

### Data Processing:
In this pipeline, all the 12 source datasets (the ones used for training the model) are loaded, preprocessed, combined, and split into train/val/test.

The Hatecheck and UNHCR datasets are also loaded and preprocessed in this pipeline.

### Model Inference:
This pipeline downloads a tokenizer and transformer model, which are used to calculate predictions.

## How to obtain the data?
All the raw data files can be downloaded from: [Download Datasets](https://stollfamily.de/01_raw.zip).

After having downloaded the data, you need to copy all the 15 folders from the `01_raw/` folder into the `data/01_raw` folder in this repo.

It is important that these datasets are kept raw as the data processing pipeline expects raw datasets as input.

To avoid writing paths to the datasets multiple places in the project, a `conf/base/catalog.yml` file is used as a form of dataset orchestrator. From the example below, it is possible to refer to `raw_unhcr` whenever you would like to provide that specific dataset as input to a node.

```
raw_unhcr:
   type: pandas.ExcelDataSet
   filepath: data/01_raw/unhcr/refugee_data_unhcr.xlsx
   layer: raw    
```

It is important to add such an instance for transformed datasets also. However, instead of storing them in `data/01_raw`, they needs to be stored in a different layer. For example, the datasets that are produced in the data processing pipeline are stored in `data/03_primary`.

## Kedro Node
When adding a node to your pipeline, it needs to have the following structure.

```python
from kedro.pipeline import node
from .nodes import run_inference

node(
    func=run_inference,
    inputs="test_unhcr",
    outputs="unhcr_predictions",
    name="run_inference_unhcr",
)
```

`func:` refers to the Python function that needs to be run.

`inputs:` the inputs that are required by the Python function. In this case, it refers to the unhcr test dataset as the function only takes a DataFrame as input.

`outputs:` the identifier that is written here needs to correspond to an instance in `conf/base/catalog.yml`.

`name:` the name of the node.

## Kedro Pipeline
The code chunk below shows how a pipeline can be constructed.
```python
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
             node(
                func=run_inference,
                inputs="test_set",
                outputs="test_set_predictions",
                name="run_inference_test_set",
             ),
             node(
                func=run_inference,
                inputs="test_hatecheck",
                outputs="hatecheck_predictions",
                name="run_inference_hatecheck",
             ),
        ]
    )
```



## How to install Kedro on a Windows Machine?

1. Create a new conda environment (IMPORTANT: the Python version must be 3.9)

2. The command below will install Kedro together with all the other project dependencies 

```bash
pip install -r src/requirements.txt
```

3. Verify the installation:

```bash
kedro info
```

## How to run your Kedro project

You can run your Kedro project with:

```bash
kedro run
```

If you want to run a specific pipeline:

```bash
kedro run -p pipeline_name
```

## Build Documentation
The documentation for this project can be built using the command below.

```bash
kedro build-docs
```
When this command has finished running, you will need to copy the path to the root folder of the project and add `docs/build/html/unhcr_hatespeech.html` to the path. This should then be inserted into your Web Browser, which will open the documentation in HTML format. 


## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r src/requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter

To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```bash
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```bash
kedro jupyter notebook
```

### JupyterLab

To use JupyterLab, you need to install it:

```bash
pip install jupyterlab
```

You can also start JupyterLab:

```bash
kedro jupyter lab
```

### IPython

And if you want to run an IPython session:

```bash
kedro ipython
```
