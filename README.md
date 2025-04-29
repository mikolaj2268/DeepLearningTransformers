# DeepLearningTransformers

## Structure
### Files:
- data_downloader.ipnyb - notebook for loading data from kaggle
- Audiodataset.py - code for processing audio files
- data_sampler.ipynb - notebook for sampling data for train dataset
- [architecture_name].ipynb - notebook for running experiments 
- plots.ipynb - notebook for plots generation
- requirements.txt - venv requirements for running the project


## Run the code

### Prerequisites

Before running the code, ensure you have the necessary libraries installed:

```bash
pip install -r requirements.txt
```

### 1. Experiments

Initialize and train the model by running architecture_name.ipynb file, for example cnn_advanced.ipnyb. The output of expewriments will be saved i hyper_results and loss_results folder. All folders will be created automatically. 


### 2. Plots

Generate plots via plots.ipnyb file, the output of the code will be saved in plots folder.

