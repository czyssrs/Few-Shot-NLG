# Few-Shot NLG
Code and data for ACL 2020 Paper "Few-Shot NLG with Pre-Trained Language Model"
https://arxiv.org/abs/1904.09521


## Installation
```
pip install -r requirements.txt
```

Due to the large consumption of GPU memory of GPT-2, we split the model into two cards and the consumption of each does not exceed 12G. 

## Instructions
Data and pre-trained GPT-2 can be downloaded via dropbox: https://www.dropbox.com/sh/u3t8yhcctqczpo0/AAAZV7S-qoIyaQW99r_88nUra?dl=0

- original: full datasets for each domain
- humans / books / songs: datasets for each domain. We provide an example processed data for 100 training examples, in preprocessed_data folder. 
- models: pre-trained GPT-2 

To get training data of other sizes, you can go to the original_data folder to sample training sets from sample_source.box and sample_source.summary, e.g., head -n 200 sample_source.box > train.box ; head -n 200 sample_source.summary > train.summary, and then run data preprocessing to generate preprocessed data. Different random samples should not make significant difference of the performances. 

Note that the experiments and results reported in the paper is on a filtered version of the original WikiBio dataset. This is because the examples in the WikiBio dataset often have information out of the input table, which is out of the scope of this few-shot learning task. Therefore we filter the dataset by a simple hueristic: set a vocabulary bound and remove the examples that have target text with oov words that's also not in in input table. 

Our method can also work on the original WikiBio dataset, the performances should drop compared to the ones on the filtered dataset due to the reasons above, but the relative improvements compared with other baselines remain still. 

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
