import os
import sys
import operator
import json
import random
import encoder
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import csv

from nipype.nipype.algorithms.icc import ICC_rep_anova
'''
some test functions
'''

def convert_bpe(box_in, summary_in, box_out, summary_out):
    """
    basically convert paranthese in stanford tokenizer
    Args:
        box_in: str, file path with info box
        summary_in: str, file path with corresponding summary
        box_out: str, output file path with info box
        summary_out: str, output file path with summary

    Returns:
        None
    """

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
    """
    join original format values
    Args:
        list_in:

    Returns:

    """

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
    """
    convert original data to bpe
    Args:
        box_in:
        summary_in:
        box_out:
        summary_out:

    Returns:

    """

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

def get_vars():
    """
    get all variable in checkpoint path
    Returns:
        None
    """
    latest_ckp = tf.train.latest_checkpoint(os.path.join('../models', '117M'))
    print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')


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




def check_encoding():
    """
    Test GPT encoder
    Returns:
        None
    """

    test1 = [10641, 829, 348, 856, 88, 357, 1248, 3270, 1377, 15533, 1267, 373]

    enc = encoder.get_encoder("117M")
    dec_out = enc.decode(test1)
    print(dec_out)


if __name__=='__main__':
    """
    Test all functions
    """
    # root_path = sys.argv[1]
    # model_path = sys.argv[2]
    # domain = sys.argv[3]


    # box_in = os.path.join(root_path, domain, 'original_data', 'test.box')
    # summary_in = os.path.join(root_path, domain, 'original_data', 'test.summary')

    # box_out = os.path.join(root_path, domain, 'original_data', 'test_full.box')
    # summary_out = os.path.join(root_path, domain, 'original_data', 'test_full.summary')

    # convert_bpe(box_in, summary_in, box_out, summary_out)

    # out_vocab = os.path.join(root_path, domain, 'original_data', 'test_vocab.txt')

    # get_train_vocab(box_in, summary_in, out_vocab)

    # json_ori_in = os.path.join(model_path, '117M_original', 'encoder.json')
    # json_out = os.path.join(model_path, '117M', 'encoder.json')
    # vocab_ind_out = os.path.join(model_path, '117M_original', 'vocab_ind.txt')
    # get_train_vocab_bpe(summary_in, box_in, json_ori_in, json_out, vocab_ind_out)

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

#   root_path = "/scratch/home/zhiyu/wiki2bio/"

#   result_language_file = root_path + "language_result.csv"
#   result_fact_file = root_path + "fact_result.csv"

#   result_fact_file = root_path + "result_fact.csv"
#   facts_score = root_path + "fact_score.txt"

#   fact_data_file = root_path + "amt_facts_data.csv"
#   fact_key_file = root_path + "amt_facts_keys.csv"


#   sum_dict_map = get_method_map(fact_data_file, fact_key_file)


#   # eval_language(result_language_file, sum_dict_map)

#   # eval_fact(result_fact_file, sum_dict_map)

#   # cal_corelation_fact(result_fact_file)

#   # cal_correlation_lan(result_language_file)

#   cal_p_fact(result_fact_file, sum_dict_map)

    check_encoding()














