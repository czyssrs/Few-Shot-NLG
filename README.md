# Few-shot NLG
Generate descriptions from wiki infobox under few shot setting

## Installation
pip install -r requirements.txt

## Instructions
There are 3 folders: code, models, and sample_data. The code folder contains all our codes and the requirement file; the models folder contains the pre-trained GPT2 model; the sample_data folder has a sample train and test data for humans domain with 100 training instances. To run our code, go to the code folder and run with: 

python ./Main.py --root_path ~/Data/NLP/few_shot_nlg/ --domain humans --gpt_model_name ../models/117M/ --output_path ~/Output/

Where the root path is the data folder. Specify an output path to store the results. The data preprocessing code can be found in preprocess.py. 

