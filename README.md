# UNHCR Hatespeech Detection
**Creators:** Frederik Gaasdal Jensen & Henry Alexander Stoll

## About this Repo
This repository is built using Kedro, which makes it possible to have a more clear structure of a project. More specifically, Kedro uses nodes and pipelines to orchestrate data flows.

***Node:*** a Python function

***Pipeline:*** a sequence of nodes

To understand the Kedro better, take a look at [Kedro documentation](https://kedro.readthedocs.io).

This project provides **two pipelines**:

1. Data Processing
2. Model Inference

### Data Processing:
In this pipeline, all the 12 source datasets (the ones used for training the model) are loaded, preprocessed, combined, and split into train/val/test.

The Hatecheck and UNHCR datasets are also loaded and preprocessed in this pipeline.

### Model Inference:
This pipeline downloads a tokenizer and transformer model, which are used to calculate predictions.


## How to install Kedro on a Windows Machine?

1. Create a new conda environment
2. Install Kedro with the following command:

```bash
conda install -c conda-forge kedro
```

3. Verify the installation:

```bash
kedro info
```

4. To visualize the pipelines, install kedro viz:

```bash
pip install kedro-viz
```

5. Install the following such that kedro works with Parquet datasets:

```bash
pip install "kedro[pandas.ParquetDataSet]"
```

## How to install dependencies

Declare any dependencies in <!-- `src/requirements.txt` for `pip` installation and  -->
`src/environment.yml` for `conda` installation.

To install them, run:

```bash
# pip install -r src/requirements.txt
conda env create -f src/environment.yml
```

TODO: To export them, run:

```bash
conda env export > src/environment.yml
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```bash
kedro run
```

If you want to run a specific pipeline:

```bash
kedro run -p pipeline_name
```

## Project Dependencies (Using Kedro)

To generate or update the dependency requirements for your project:

```bash
kedro build-reqs
```

This will `pip-compile` the contents of `src/requirements.txt` into a new file `src/requirements.lock`. You can see the output of the resolution by opening `src/requirements.lock`.

After this, if you'd like to update your project requirements, please update `src/requirements.txt` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## Project Dependencies (Without using Kedro)
Another way to do this without Kedro is with the following command. 

```bash
pip install -r src/requirements.txt
```

## Build Documentation
The documentation for this project can be built using the below command.

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



## TODOs

* [ ] push dataset to huggingface
* [ ] pull model from hugginface and run inference
  * [ ] hatecheck
  * [ ] unhcr
* [ ] train model: Henry
* [ ] vizualize hatecheck (somehow) -> html / viz
* [ ] documentation
  * [ ] README
  * [ ] documentation
  * [ ] presentation?
