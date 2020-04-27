# Few-Shot NLG
Code and data for paper ACL 2020 Paper "Few-Shot NLG with Pre-Trained Language Model"
https://arxiv.org/abs/1904.09521


## Installation
```
pip install -r requirements.txt
```

## Instructions
Data and pre-trained GPT-2 can be downloaded via dropbox: https://www.dropbox.com/sh/u3t8yhcctqczpo0/AAAZV7S-qoIyaQW99r_88nUra?dl=0
```
-- data_release
---- original: full datasets for each domain
---- humans / books / songs: datasets for each domain. We provide an example processed data for 100 training examples, in preprocessed_data folder, that you can directly train the model with. To get training data of other sizes, you can go to the original_data folder to sample training sets from sample_source.box and sample_source.summary, e.g., head -n 200 sample_source.box > train.box ; head -n 200 sample_source.summary > train.summary, and then run data preprocessing to generate preprocessed data.
-- models: pre-trained GPT-2 
```
To run our code, go to the code folder and run with: 

Data preprocessing:
```
python preprocess.py ~/Data/NLP/few_shot_nlg/ humans
```
Training:
```
python ./Main.py --root_path ~/Data/NLP/few_shot_nlg/ --domain humans --gpt_model_name ../models/117M/ --output_path ~/Output/
```
Where the root path is the data folder. Specify an output path to store the results. 

If you find our work helpful, please cite the arxiv version. 

