# Supervised Learning apporach for DNA classification - Deep Learning
Using Frequency matrix of Chaos Game Representation (FCGR) as input.

**GOAL** Classify intra-specie DNA sequences in a supervised fashion. 

The codes are organized as follows: 

1. Set the parameters for the experiment in `parameters.py`: 
Data to use:  
- Specie (Bacteria, Dengue, Fish, etc)
- $Kmer$ (FCGR of dimension $(2^{Kmer}, 2^{Kmer})$)
- Batch size, train size, number of epochs 

2. Generate inputs (images), train and test models: 

- `01-generate_fcgr.py`: Generates the FCGR images for sequences in `/data` folder (from [DeLUCS](https://github.com/millanp95/DeLUCS/tree/master/data)), and put them in a new folder called `/img-{KMER}-mer`
- `02-train_val_test_split.py`: Split the generated images in  train, validation and test sets. They are balanced based on labels. Results of the selected sets are saved in `datasets.json`. A summary of the selected labels can be see in `summary_labels.csv`
- `03-train_model.py`: train the model. Best weights are saved at `/checkpoint` folder. 
- `04-test_model.py`: evaluation of the model in the test set. Metrics of the test set are saved in `metrics_test.csv`
 
*Images and sequences are saved (and tracked with GIT) in google drive using [DVC](dvc.org)*