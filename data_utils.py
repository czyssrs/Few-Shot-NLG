import sys
import os
import operator
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
import json
import zipfile
import string
import queue
import random
import encoder
import numpy as np
from fuzzysearch import find_near_matches
from tqdm import tqdm

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def convert_bpe(box_in, summary_in, box_out, summary_out):
	'''
	basically convert paranthese in stanford tokenizer
	'''

	with open(box_in) as f:
		lines_box = f.readlines()

	with open(summary_in) as f:
		lines_summary = f.readlines()

	out_b = open(box_out, "w")
	out_s = open(summary_out, "w")

	for line_box, line_summary in zip(lines_box, lines_summary):
		line_box = line_box.replace("-lrb-", "(")
		line_box = line_box.replace("-rrb-", ")")

		line_summary = line_summary.replace("-lrb-", "(")
		line_summary = line_summary.replace("-rrb-", ")")

		out_b.write(line_box)
		out_s.write(line_summary)



	out_b.close()
	out_s.close()




def join_box(list_in):
	'''
	join original format values
	'''

	out_list = []
	current_name = ""
	current_value = ""
	# print "\n"
	# print list_in

	for each_item in list_in:
		field_name = each_item.split(":")[0]
		field_value = each_item.split(":")[1]

		if field_name == "":
			continue

		if not field_name[-1].isdigit():
			if field_value != "<none>":
				out_list.append((field_name, field_value))
			continue

		field_name = "_".join(field_name.split("_")[:-1])

		if field_name != current_name:
			if current_name != "":
				cur_name_list = [tup[0] for tup in out_list]
				# print out_list
				# print field_name
				# assert field_name not in cur_name_list

				### remove none value
				if current_value.strip() != "<none>":
					out_list.append((current_name, current_value.strip()))
				current_name = ""
				current_value = ""

		current_name = field_name
		current_value += (field_value + " ")


	if current_value.strip() != "<none>":
		out_list.append((current_name, current_value.strip()))

	sorted_by_second = sorted(out_list, key=lambda tup: len(tup[1].split(" ")), reverse=True)

	random_out = random.shuffle(sorted_by_second)

	return out_list, sorted_by_second




def gen_bpe_data(box_in, summary_in, box_out, summary_out):
	'''
	convert original data to bpe
	'''

	enc = encoder.get_encoder("117M")

	with open(box_in) as f:
		lines_box = f.readlines()

	with open(summary_in) as f:
		lines_summary = f.readlines()

	out_b = open(box_out, "w")
	out_s = open(summary_out, "w")

	for line_summary, line_box in tqdm(zip(lines_summary, lines_box)):

		this_summary = line_summary.strip()
		### stanford tokenizer not apply here
		this_summary = this_summary.replace("-lrb-", "(")
		this_summary = this_summary.replace("-rrb-", ")")
		# this_summary = " " + this_summary

		tokens_summary, tokens_original = enc.encode(this_summary)

		this_out_summary = " ".join(tokens_original)

		box_list = line_box.strip().split("\t")
		box_out_list, box_field_list = join_box(box_list)

		this_out_box_list = []

		for field_name, field_value in box_out_list:

			if field_name != "name":
				field_value = " " + field_value

				field_value = field_value.replace("-lrb-", "(")
				field_value = field_value.replace("-rrb-", ")")

			# field_value = " " + field_value

			tokens, tokens_original = enc.encode(field_value)

			for ind, each_token in enumerate(tokens_original):

				this_out_box_list.append(field_name + "_" + str(ind + 1) + ":" + each_token)






		# print (line)
		# print (tokens)
		# print (tokens_original)
		# print (len(line.split()))
		# print (len(tokens))
		# print ("\n")


		out_s.write(this_out_summary + "\n")
		out_b.write("\t".join(this_out_box_list) + "\n")


		# print (this_summary)
		# print (this_out_summary)
		# print (tokens_summary)
		# print ("\t".join(this_out_box_list))
		# print ("\n")




	out_b.close()
	out_s.close()



def test():

	test1 = " name"
	# test1 = [10641, 829, 348, 856, 88, 357, 1248, 3270, 1377, 15533, 1267, 373, 281, 46932, 16716, 10099, 290, 1772, 764]
	test1 = [10641, 829, 348, 856, 88, 1149, 829, 348, 856, 88, 1248, 3270, 12, 1129, 1270, 764]
	# test1 = [889, 1289]


	enc = encoder.get_encoder("117M")
	# tokens, tokens_original = enc.encode(test1)

	dec_out = enc.decode(test1)
	print (dec_out)

	# print (" ".join(tokens_original))
	# print (tokens)
	# print (tokens)


def get_vars():
	'''
	get all variable in checkpoint path
	'''


	latest_ckp = tf.train.latest_checkpoint(os.path.join('../models', '117M'))
	print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')


def get_train_vocab(box_in, summary_in, out_vocab):
	'''
	get vocab for few shot baselines
	'''

	vocab = {}

	with open(box_in) as f:
		for line in f:
			line_list = line.strip().split()
			for item in line_list:
				if ":" in item:
					field = item.split(":")[0]
					value = item.split(":")[1]

					if field != "" and value != "":
						if "_" in field:
							field = field.split("_")[0]

						if field not in vocab:
							vocab[field] = 0
						vocab[field] += 1

						if value not in vocab:
							vocab[value] = 0
						vocab[value] += 1



	with open(summary_in) as f:
		for line in f:
			line_list = line.strip().split()
			for word in line_list:
				if word not in vocab:
					vocab[word] = 0
				vocab[word] += 1


	sorted_x = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)


	ind = 0
	with open(out_vocab, "w") as f:
		for tup in sorted_x:
			if tup[1] > 0:
				f.write(tup[0] + "\t" + str(ind) + "\n")
				ind += 1


	print (len(sorted_x))
	print (ind)



def get_train_vocab_bpe(summary_in, out_vocab):
	'''
	get train vocab of gpt data. return the mask
	'''

	vocab = []
	enc = encoder.get_encoder("117M")
	vocab_len = 50257

	with open(summary_in) as f:
		for line in f:
			line = line.strip()
			tokens, tokens_original = enc.encode(line)

			for token in tokens:
				if token not in vocab:
					vocab.append(token)
			

	print (len(vocab))


	res_mask = []
	for ind in range(0, 50257):
		if ind in vocab:
			res_mask.append(str(1))
		else:
			res_mask.append(str(0))


	with open(out_vocab, "w") as f:
		f.write(" ".join(res_mask))










if __name__=='__main__':

	# box_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data_original/test.box"
	# summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data_original/test.summary"
	# box_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/test.box"
	# summary_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/test.summary"
	# gen_bpe_data(box_in, summary_in, box_out, summary_out)

	# test()

	# box_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans_original/original_data_original/test_full.box"
	# summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans_original/original_data_original/test_full.summary"

	# box_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans_original/original_data/test_full.box"
	# summary_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans_original/original_data/test_full.summary"

	# convert_bpe(box_in, summary_in, box_out, summary_out)

	# enc = encoder.get_encoder("117M")
	# print(enc.encoder['<|endoftext|>'])

	# get_vars()



	# ### generate vocab for baseline
	# box_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt_baseline_emb_pointer/humans/original_data/train_200.box"
	# summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt_baseline_emb_pointer/humans/original_data/train_200.summary"
	# out_vocab = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt_baseline_emb_pointer/humans_vocab_200.txt"
	# get_train_vocab(box_in, summary_in, out_vocab)


	### generate mask for gpt
	summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/train_200.summary"
	out_vocab = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/vocab_200.txt"
	get_train_vocab_bpe(summary_in, out_vocab)















