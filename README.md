# Few-shot table-to-text Generation
Generate descriptions from wiki infobox under few shot setting

## Installation
python3 -m venv ucsb
source ucsb/bin/activate
pip install -r requirements.txt

## Instructions
Download the data(few_shot_gpt-2.zip) and pre-trained GPT2 model(models.zip), unzip them to the same directory of the code(so that all three of them(few_shot_gpt-2_data, models, wikitobio) are in the same directory).

The training takes two 10G GPUs. I placed the GPT part into one GPU and the pointer generator part into another, since on my side we only have 10G GPUs. If you have different GPU configurations, like one GPU of more than 20G, go to SeqUnit.py and remove all the "with tf.device("/gpu:1"):" tags. 

Command for training:

$ cd wikitobio

$ python Main.py your_saved_model_name few_shot_folder

Now it's the experiment on humans domain with 1000 training data. You will see a result bleu score of around 26 after 10 training rounds. 
