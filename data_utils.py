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
	# 10641 829 348 856 88 357 1248 3270 1377 15533 1267 373 281 46932 16716 10099 290 1772 764
	# test1 = [12427, 17266, 32790, 72, 38325]
	# test1 = [35626, 266, 13, 6314, 2162, 4082, 3128, 1058, 1511, 46593, 346, 24217, 2162, 29835, 1058, 45630, \
	# 272, 2162, 9901, 1058, 410, 4066, 34508, 6403, 2162, 2214, 1058, 3230, 12446, 2162, 435, 2611, 26910, 1058, \
	# 6403, 286, 686, 35983, 2162, 3172, 66, 21231, 1058, 277, 2162, 3172, 66, 4686, 1058, 279, 2127, 22515, 2162, 2708, 3670, 1058, 1931, 291, 6314, 2162]
	# test1 = [3672, 837, 2939, 837, 8305, 837, 13755, 837, 4082, 3128, 837, 4082, 1295, 837, 1918, 3128, 837, 1918, 1295, 837, 16731, 837, 3397, 837, 2708, 3670, 837]
	test1 = [7839, 1058, 4171, 2933, 837, 1772, 1058, 474, 11025, 308, 295, 78, 837, 2656, 3670 ,1058, 474 ,11025, 443, 7245, 84 ,837 ,33417, 1058, 479 ,15289,257 ,13 ,10212, 365 ,837 ,1499 ,\
	1058 ,1216, 590 ,837, 3303, 1058, 48718, 837 ,9991 ,1058 ,38251, 67 ,1756 ,8701, 316, 837 ,9207, 3128, 1058,32471, 837, 3199 ,287 ,46932, 1058 ,22717 ,837, 5468 ,1058 ,34131, 837]
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



def get_train_vocab_bpe_mask(summary_in, out_vocab):
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



def get_train_vocab_bpe(summary_in, box_in, json_ori_in, json_out, vocab_ind_out):
	'''
	get train vocab of gpt data. return the mask
	'''

	vocab = []
	enc = encoder.get_encoder("117M_original")
	vocab_len = 50257

	with open(summary_in) as f:
		for line in f:
			line = line.strip()
			tokens, tokens_original = enc.encode(line)

			for token in tokens:
				if token not in vocab:
					vocab.append(token)
			

	with open(box_in) as f:
		for line in f:
			line_list = line.strip().split("\t")

			out_list, sorted_by_second = join_box(line_list)

			for (this_name, this_value) in out_list:

				bpe_in = " " + this_name.replace("_", " ")

				tokens, tokens_original = enc.encode(bpe_in)

				for token in tokens:
					if token not in vocab:
						vocab.append(token)


				if this_name != "name":
					bpe_in = " " + this_value
				else:
					bpe_in = this_value


				tokens, tokens_original = enc.encode(bpe_in)

				for token in tokens:
					if token not in vocab:
						vocab.append(token)

	print (len(vocab))


	res_vocab = []
	for ind in range(0, 50257):
		if ind < 100:
			res_vocab.append(ind)
		elif ind in vocab:
			res_vocab.append(ind)
		elif ind == 28920:
			res_vocab.append(ind)
		elif ind == 50256:
			res_vocab.append(ind)


	# if 28920 not in res_vocab:
	# 	res_vocab.append(28920)

	# if 50256 not in res_vocab:
	# 	res_vocab.append(50256)

	with open(json_ori_in) as f:
		tmp = f.readline().strip()
		vocab_tmp = json.loads(tmp)

	vocab_ori = {value: key for key, value in vocab_tmp.items()}

	out_vocab = {}
	for ind_new, ind in enumerate(res_vocab):
		token = vocab_ori[ind]
		out_vocab[token] = ind_new
	
	print (len(out_vocab))
	print (out_vocab["empty"])
	print (out_vocab["<|endoftext|>"])


	with open(json_out, "w") as f:
		f.write(json.dumps(out_vocab))

	with open(vocab_ind_out, "w") as f:
		for ind in res_vocab:
			f.write(str(ind) + "\n")



if __name__=='__main__':

	# box_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data_original/test.box"
	# summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data_original/test.summary"
	# box_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/test.box"
	# summary_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/test.summary"
	# gen_bpe_data(box_in, summary_in, box_out, summary_out)

	# test()

	box_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/songs/original_data_original/test.box"
	summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/songs/original_data_original/test.summary"

	box_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/songs/original_data/test_full.box"
	summary_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/songs/original_data/test_full.summary"

	convert_bpe(box_in, summary_in, box_out, summary_out)

	# enc = encoder.get_encoder("117M")
	# print(enc.encoder['<|endoftext|>'])

	# get_vars()



	# ### generate vocab for baseline
	# box_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt_baseline_emb_pointer/humans/original_data/train_1000.box"
	# summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt_baseline_emb_pointer/humans/original_data/train_1000.summary"
	# out_vocab = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt_baseline_emb_pointer/humans_vocab_1000.txt"
	# get_train_vocab(box_in, summary_in, out_vocab)


	# ### generate mask for gpt
	# summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans_tune/original_data/train_1000.summary"
	# box_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans_tune/original_data/train_1000.box"
	# json_ori_in = "/scratch/home/zhiyu/wiki2bio/models/117M_original/encoder.json"
	# vocab_ind_out = "/scratch/home/zhiyu/wiki2bio/models/117M/vocab_ind.txt"
	# json_out = "/scratch/home/zhiyu/wiki2bio/models/117M/encoder.json"
	# get_train_vocab_bpe(summary_in, box_in, json_ori_in, json_out, vocab_ind_out)















