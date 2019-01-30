import sys
import os
from nltk.translate.bleu_score import sentence_bleu


project_dir = "/scratch/home/zhiyu/wiki2bio/"
model_dir = project_dir + "results/res/pointer-gen/loads/33/"

train_table = project_dir + "original_data/train.box"
test_table = project_dir + "original_data/test.box"
summary_gold_in = project_dir + "original_data/test.summary"

summary_ours_in = model_dir + "test_summary_copy.clean.txt"
summary_with_unk = model_dir + "test_summary_copy.txt"

file_out = project_dir + "data_misc/test_compare_error_0.5_pointer.txt"

def gen_compare(table_in, summary_ours_in, summary_with_unk, summary_gold_in, file_out):
	'''
	gen file for error case compare
	'''

	with open(table_in) as f:
		table = f.readlines()

	with open(summary_ours_in) as f:
		res_ours = f.readlines()

	with open(summary_with_unk) as f:
		res_ours_unk = f.readlines()

	with open(summary_gold_in) as f:
		res_gold = f.readlines()

	out = open(file_out, "w")
	i = 0
	num_write = 0
	avg_len_right = 0.0
	avg_len_wrong = 0.0

	for this_box, this_ours, this_unk, this_gold in zip(table, res_ours, res_ours_unk, res_gold):

		references = [this_gold.strip().split()]
		hypothesis = this_ours.split()
		this_bleu = sentence_bleu(references, hypothesis)
		i += 1

		#if this_bleu < 0.1:
		#if this_bleu > 0.1 and this_bleu < 0.2:
		#if this_bleu > 0.2 and this_bleu < 0.3:
		if this_bleu > 0.5:

			avg_len_wrong += len(this_box.strip().split("\t"))
			num_write += 1
			out.write("########## Test " + str(i) +  " ##########\n")

			for each_token in this_box.strip().split("\t"):
				out.write(each_token + "\n")

			out.write("########## Gold ##########\n")
			out.write(this_gold.strip() + "\n")
			out.write("########## Ours ##########\n")
			out.write(this_ours.strip() + "\n")
			out.write("########## Ours with unk ##########\n")
			out.write(this_unk.strip() + "\n")
			out.write("########## bleu ##########\n")
			out.write(str(this_bleu) + "\n")
			out.write("\n")

		else:
			avg_len_right += len(this_box.strip().split("\t"))


	out.close()

	print "All: ", i
	print "Write: ", num_write

	print "avg_len_wrong: ", float(avg_len_wrong) / num_write
	print "avg_len_right: ", float(avg_len_right) / (i - num_write)



def check_field(table_in, field_name):


	with open(table_in) as f:
		table = f.readlines()

	num_all = 0
	num_target = 0

	for this_box in table:

		num_all += 1

		for each_token in this_box.strip().split("\t"):

			if each_token.split(":")[0].split("_")[0] == field_name:
				num_target += 1
				break



	print "All: ", num_all
	print "Target: ", num_target
	print float(num_target) / num_all




if __name__=='__main__':

	gen_compare(test_table, summary_ours_in, summary_with_unk, summary_gold_in, file_out)
	#check_field(train_table, "succession")



























