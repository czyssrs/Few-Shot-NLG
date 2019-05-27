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
import csv

from nipype.nipype.algorithms.icc import ICC_rep_anova


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


def convert_bpe_inv(box_in, summary_in, box_out, summary_out):

	with open(box_in) as f:
		lines_box = f.readlines()

	with open(summary_in) as f:
		lines_summary = f.readlines()

	out_b = open(box_out, "w")
	out_s = open(summary_out, "w")

	for line_box, line_summary in zip(lines_box, lines_summary):
		line_box = line_box.replace("(", "-lrb-")
		line_box = line_box.replace(")", "-rrb-")

		line_summary = line_summary.replace("(", "-lrb-")
		line_summary = line_summary.replace(")", "-rrb-")

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



def gen_case(gold_box, gold_summary, path_original, path_switch, path_gpt_only, path_ours, out):

	'''
	generate case study
	'''

	with open(gold_box) as f:
		gb = f.readlines()

	with open(gold_summary) as f:
		gs = f.readlines()


	with open(path_original) as f:
		original = f.readlines()

	with open(path_switch) as f:
		switch = f.readlines()


	with open(path_gpt_only) as f:
		gpt_only = f.readlines()


	with open(path_ours) as f:
		ours = f.readlines()



	ind = 0
	with open(out, "w") as f:
		for box_gold, summary_gold, s_original, s_switch, s_gpt_only, s_ours in zip(gb, gs, original, switch, gpt_only, ours):
			f.write("\n")
			f.write("###################################\n")

			box_in = box_gold.strip().split("\t")
			box_list, _ = join_box(box_in)

			box_write = ""
			for tup in box_list:
				box_write += (tup[0] + " : ")
				box_write += (tup[1] + ";\t")

			f.write(box_write)
			f.write("\n")
			f.write("Gold: \n")
			f.write(summary_gold)
			f.write("FG: \n")
			f.write(s_original)
			f.write("Switch: \n")
			f.write(s_switch.strip() + "\n")
			f.write("GPT only: \n")
			f.write(s_gpt_only.strip() + "\n")
			f.write("Ours: \n")
			f.write(s_ours.strip() + "\n")

			# ind += 1
			# if ind > 499:
			# 	break



def gen_csv_fact(gold_box, gold_summary, path_switch, path_gpt_only, path_ours, out):
	'''
	generate csv for AMT.
	counting facts
	'''

	import csv

	with open(gold_box) as f:
		gb = f.readlines()

	with open(gold_summary) as f:
		gs = f.readlines()


	with open(path_switch) as f:
		switch = f.readlines()


	with open(path_gpt_only) as f:
		gpt_only = f.readlines()


	with open(path_ours) as f:
		ours = f.readlines()




	with open(out, "w") as f:

		w = csv.writer(f)
		w.writerow(['Ind', 'Method', 'Table', 'Text'])

		ind = 0
		for box_gold, summary_gold, s_switch, s_gpt_only, s_ours in zip(gb, gs, switch, gpt_only, ours):

			box_in = box_gold.strip().split("\t")
			box_list, _ = join_box(box_in)

			box_write = ""
			for tup in box_list:
				box_write += (tup[0] + " : ")
				box_write += (tup[1] + ";\t")

			# f.write(box_write)
			# f.write("\n")
			# f.write("Gold: \n")
			# f.write(summary_gold)
			# f.write("Switch: \n")
			# f.write(s_switch)
			# f.write("GPT only: \n")
			# f.write(s_gpt_only)
			# f.write("Ours: \n")
			# f.write(s_ours)

			# Method = "gold"
			w.writerow([str(ind), "gold", box_write, summary_gold])

			# switch
			w.writerow([str(ind), "switch", box_write, s_switch])

			# gpt only
			w.writerow([str(ind), "w/o_copy", box_write, s_gpt_only])

			# ours
			w.writerow([str(ind), "ours", box_write, s_ours])

			ind += 1

			if ind > 499:
				break



def eval_fact_avg_score(file_in, file_out):
	'''
	get average score for sup and contra 
	each summary as key
	'''

	num_approved = 0
	num_summary = 0

	### {summary: [sup, contra, times]}
	summary_dict = {}

	with open(file_in) as f:
		f_csv = csv.DictReader(f)
		for row in f_csv:
			# print (row['Input.summary'])
			# print (row['AssignmentStatus'])

			if row['AssignmentStatus'] == "Approved":
				num_approved += 1
				summary = row['Input.summary']
				sup = int(row['Answer.support'])
				contra = int(row['Answer.contra'])


				if summary not in summary_dict:
					num_summary += 1
					summary_dict[summary] = [sup, contra, 1]

				else:
					summary_dict[summary][0] += sup
					summary_dict[summary][1] += contra
					summary_dict[summary][2] += 1


	check_dict = {}
	for summary in summary_dict:
		times = summary_dict[summary][2]
		if times not in check_dict:
			check_dict[times] = 0
		check_dict[times] += 1


	print (num_summary)
	print (check_dict)
	# print (num_approved)




def get_method_map(file_data, file_key):
	'''
	get the method of each summary
	'''

	sum_list = []
	with open(file_data) as f:
		f_csv = csv.DictReader(f)
		for row in f_csv:
			sum_list.append(row['summary'])


	# print (len(sum_list))

	label_list = []
	with open(file_key) as f:
		f_csv = csv.DictReader(f)
		for row in f_csv:
			label_list.append(row['key'])

	# print (len(label_list))


	sum_dict_map = {}

	for summary, label in zip(sum_list, label_list):
		if summary not in sum_dict_map:
			sum_dict_map[summary] = []

		if label in sum_dict_map[summary]:
			print ("error!")

		sum_dict_map[summary].append(label)

	# check_dict = {}
	# for summary in sum_dict_map:
	# 	times = len(sum_dict_map[summary])
	# 	if times == 3:
	# 		print (summary)
	# 	if times not in check_dict:
	# 		check_dict[times] = 0
	# 	check_dict[times] += 1

	# print (check_dict)


	# print (sum_dict_map)
	return sum_dict_map


def eval_language(file_in, sum_dict_map):
	'''
	evaluate language fluency
	'''

	num = 0

	scores = {'gold':0.0, 'gpt':0.0, 'switch':0.0, 'ours':0.0}

	with open(file_in) as f:
		f_csv = csv.DictReader(f)
		for row in f_csv:
			if row['AssignmentStatus'] == "Approved":
				answer = row['Answer.grammaticality.label']
				summary1 = row['Input.text1']
				summary2 = row['Input.text2']

				if answer == "1":
					method_list = sum_dict_map[summary1]
					this_score = (1.0 / len(method_list))
					for each_method in method_list:
						scores[each_method] += this_score


				elif answer == "2":
					method_list = sum_dict_map[summary2]
					this_score = (1.0 / len(method_list))
					for each_method in method_list:
						scores[each_method] += this_score

				elif answer == "Tied":
					method_list_1 = sum_dict_map[summary1]
					this_score_1 = (0.5 / len(method_list_1))
					for each_method in method_list_1:
						scores[each_method] += this_score_1

					method_list_2 = sum_dict_map[summary2]
					this_score_2 = (0.5 / len(method_list_2))
					for each_method in method_list_2:
						scores[each_method] += this_score_2


	for method in scores:
		scores[method] /= (3.0 * 499)

	print (scores)


def eval_fact(file_in, sum_dict_map):
	'''
	evaluate facts
	'''

	num = 0

	## summary: [sup, con, times]
	sum_dict_counter = {}

	scores_sup = {'gold':0.0, 'gpt':0.0, 'switch':0.0, 'ours':0.0}
	scores_con = {'gold':0.0, 'gpt':0.0, 'switch':0.0, 'ours':0.0}
	score_times = {'gold':0.0, 'gpt':0.0, 'switch':0.0, 'ours':0.0}

	with open(file_in) as f:
		f_csv = csv.DictReader(f)
		for row in f_csv:
			if row['AssignmentStatus'] == "Approved":
				summary = row['Input.summary']
				sup = int(row['Answer.support'])
				con = int(row['Answer.contra'])

				if summary not in sum_dict_counter:
					sum_dict_counter[summary] = [sup, con, 1.0]
				else:
					sum_dict_counter[summary][0] += sup
					sum_dict_counter[summary][1] += con
					sum_dict_counter[summary][2] += 1



	for summary in sum_dict_counter:
		sup = sum_dict_counter[summary][0]
		con = sum_dict_counter[summary][1]
		times = sum_dict_counter[summary][2]
		sup_avg = float(sup) / times
		con_avg = float(con) / times

		method_list = sum_dict_map[summary]
		for method in method_list:
			scores_sup[method] += sup_avg
			scores_con[method] += con_avg
			score_times[method] += 1



	for method in scores_sup:
		scores_sup[method] /= 499
		scores_con[method] /= 499


	print (scores_sup)
	print (scores_con)
	print (score_times)


def cal_corelation_fact(file_in):
	'''
	calculate correlation for fact eval
	'''

	res_dict_sup = {}
	res_dict_con = {}


	with open(file_in) as f:
		f_csv = csv.DictReader(f)
		for row in f_csv:
			if row['AssignmentStatus'] == "Approved":
				# hitid = row['HITId']
				hitid = row['Input.summary']
				summary = row['Input.summary']
				sup = int(row['Answer.support'])
				con = int(row['Answer.contra'])

				if hitid not in res_dict_sup:
					res_dict_sup[hitid] = []
					res_dict_con[hitid] = []

				res_dict_sup[hitid].append(sup)
				res_dict_con[hitid].append(con)




	# times_map = {}

	# for item in res_dict:
	# 	if res_dict[item] not in times_map:
	# 		times_map[res_dict[item]] = 0
	# 	times_map[res_dict[item]] += 1


	pass_icc_sup = []

	for item in res_dict_sup:
		this_line = res_dict_sup[item]
		if len(this_line) == 3:
			pass_icc_sup.append(this_line)

	pass_icc_con = []

	for item in res_dict_con:
		this_line = res_dict_con[item]
		if len(this_line) == 3:
			# print (this_line)
			pass_icc_con.append(this_line)


	# print (len(pass_icc_sup))
	# print (len(pass_icc_con))

	# pass_icc_con = [[1,1,1], [1,2,1], [1,1,2]]

	icc, *tail = ICC_rep_anova(np.array(pass_icc_sup))

	print (icc)


def fleiss_kappa(M):
  """
  See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
  :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
  :type M: numpy matrix
  """
  N, k = M.shape  # N is # of items, k is # of categories
  n_annotators = float(np.sum(M[0, :]))  # # of annotators

  p = np.sum(M, axis=0) / (N * n_annotators)
  P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
  Pbar = np.sum(P) / N
  PbarE = np.sum(p * p)

  kappa = (Pbar - PbarE) / (1 - PbarE)

  return kappa


def cal_correlation_lan(file_in):
	'''
	calculate fleiss for language score
	'''

	res_dict = {}


	with open(file_in) as f:
		f_csv = csv.DictReader(f)
		for row in f_csv:
			if row['AssignmentStatus'] == "Approved":
				answer = row['Answer.grammaticality.label']
				summary1 = row['Input.text1']
				summary2 = row['Input.text2']
				hitid = row['HITId']

				if hitid not in res_dict:
					res_dict[hitid] = []

				res_dict[hitid].append(answer)



	# times_map = {}

	# for item in res_dict:
	# 	if res_dict[item] not in times_map:
	# 		times_map[res_dict[item]] = 0
	# 	times_map[res_dict[item]] += 1


	res_mat = []

	for hitid in res_dict:

		if len(res_dict[hitid]) == 3:
			this_res = [0,0,0]
			for each_ans in res_dict[hitid]:
				if each_ans == "1":
					this_res[0] += 1
				elif each_ans == "2":
					this_res[1] += 1
				elif each_ans == "Tied":
					this_res[2] += 1


				else:
					print ("label error")


			# print (this_res)
			res_mat.append(this_res)

	# print (len(res_mat))

	# kappa = fleiss_kappa(np.array(res_mat))


	num_3 = 0
	num_2 = 0
	for item in res_mat:
		if 3 in item:
			num_3 += 1
		elif 2 in item:
			num_2 += 1


	print (float(num_3) / len(res_mat))
	print (float(num_2) / len(res_mat))

	# print (kappa)


def cal_p_fact(file_in, sum_dict_map):
	'''
	p_value test
	'''

	from statsmodels.stats.multicomp import pairwise_tukeyhsd
	from statsmodels.stats.multicomp import MultiComparison

	sup_dict = {}
	con_dict = {}

	with open(file_in) as f:
		f_csv = csv.DictReader(f)
		for row in f_csv:
			if row['AssignmentStatus'] == "Approved":
				summary = row['Input.summary']
				sup = int(row['Answer.support'])
				con = int(row['Answer.contra'])


				if summary not in sup_dict:
					sup_dict[summary] = []
					con_dict[summary] = []

				sup_dict[summary].append(sup)
				con_dict[summary].append(con)


	res_sup_method = []
	res_sup_score = []

	for summary in sup_dict:
		if len(sup_dict[summary]) == 3:
			method_list = sum_dict_map[summary]
			score_list = sup_dict[summary]
			for method in method_list:
				for score in score_list:
					res_sup_method.append(method)
					res_sup_score.append(score)

	print (len(res_sup_method))


	mc = MultiComparison(res_sup_score, res_sup_method)
	result = mc.tukeyhsd(0.01)
	 
	print(result)
	print(mc.groupsunique)






if __name__=='__main__':


	root_path = "/scratch/home/zhiyu/wiki2bio/"

	result_language_file = root_path + "language_result.csv"
	result_fact_file = root_path + "fact_result.csv"

	result_fact_file = root_path + "result_fact.csv"
	facts_score = root_path + "fact_score.txt"

	fact_data_file = root_path + "amt_facts_data.csv"
	fact_key_file = root_path + "amt_facts_keys.csv"


	sum_dict_map = get_method_map(fact_data_file, fact_key_file)


	# eval_language(result_language_file, sum_dict_map)

	# eval_fact(result_fact_file, sum_dict_map)

	# cal_corelation_fact(result_fact_file)

	# cal_correlation_lan(result_language_file)

	cal_p_fact(result_fact_file, sum_dict_map)






	# root_path = "/scratch/home/zhiyu/wiki2bio/"
	# gold_box = root_path + "few_shot_gpt-2/humans/original_data/test.box"
	# gold_summary = root_path + "few_shot_gpt-2/humans/original_data/test.summary"
	# path_original = root_path + "few_shot_gpt_baseline_original_subword/humans/results/res/res_200/loads/32/valid_summary.clean.txt"
	# path_switch = root_path + "few_shot_gpt_baseline_emb_pointer_subword/humans/results/res/res_200/loads/39/valid_summary.clean.txt"
	# path_ours = root_path + "few_shot_gpt-2/humans/results/res/res_200_copy0.5/loads/7/valid_summary.clean.txt"
	# path_gpt_only = root_path + "few_shot_gpt-2/humans/results/res/res_200_real/loads/3/valid_summary.clean.txt"

	# out = root_path + "manual_result.txt"


	# gen_case(gold_box, gold_summary, path_original, path_switch, path_gpt_only, path_ours, out)


	# gen_case(gold_box, gold_summary, path_switch, path_gpt_only, path_ours, out)

	# box_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/train_full_prev.box"
	# summary_in = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/train_full_prev.summary"
	# box_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/train_full.box"
	# summary_out = "/scratch/home/zhiyu/wiki2bio/few_shot_gpt-2/humans/original_data/train_full.summary"
	# # gen_bpe_data(box_in, summary_in, box_out, summary_out)

	# # test()

	# # box_in = "/scratch/home/zhiyu/wiki2bio/wiki2bio/original_data/test_old.box"
	# # summary_in = "/scratch/home/zhiyu/wiki2bio/wiki2bio/original_data/test_old.summary"

	# # box_out = "/scratch/home/zhiyu/wiki2bio/wiki2bio/original_data/test.box"
	# # summary_out = "/scratch/home/zhiyu/wiki2bio/wiki2bio/original_data/test.summary"

	# convert_bpe(box_in, summary_in, box_out, summary_out)

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















