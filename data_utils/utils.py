import sys
import os
import operator
from nltk.translate.bleu_score import sentence_bleu
import json
from nlgeval import NLGEval
from nlgeval import compute_metrics


project_dir = "/scratch/home/zhiyu/wiki2bio/"
model_dir = project_dir + "results/res/pointer-gen/loads/33/"
#model_dir = "/scratch/home/zhiyu/wiki2bio_ori/results/res/faget/1548416477366/loads/32/"

train_table = project_dir + "original_data/train.box"
test_table = project_dir + "original_data/test.box"
summary_gold_in = project_dir + "original_data/test.summary"

summary_ours_in = model_dir + "test_summary_copy.clean.txt"
summary_with_unk = model_dir + "test_summary_copy.txt"

### evaluate
cider_in = project_dir + "cider/results-baseline.json"

file_out = project_dir + "data_misc/test_compare_error_cider_1.0_baseline.txt"

def gen_compare(cider_in, table_in, summary_ours_in, summary_with_unk, summary_gold_in, file_out):
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

	with open(cider_in) as f:
		res_cider = json.load(f)
		score_cider = res_cider["CIDEr"]
		score_cider_d = res_cider["CIDErD"]

		print len(score_cider_d)

	nlgeval = NLGEval()

	out = open(file_out, "w")
	i = 0
	num_write = 0
	avg_len_right = 0.0
	avg_len_wrong = 0.0

	for this_cider, this_box, this_ours, this_unk, this_gold in zip(score_cider_d, table, res_ours, res_ours_unk, res_gold):

		references = [this_gold.strip().split()]
		hypothesis = this_ours.strip().split()
		this_bleu = sentence_bleu(references, hypothesis)
		this_cider = float(this_cider)
		metrics_dict = nlgeval.compute_individual_metrics([this_gold.strip()], this_ours.strip())
		this_meteor = metrics_dict["meteor"]

		i += 1

		#if this_bleu < 0.1:
		#if this_bleu > 0.1 and this_bleu < 0.2:
		#if this_bleu > 0.2 and this_bleu < 0.3:
		if this_cider < 1.0:

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
			out.write("########## cider-d ##########\n")
			out.write(str(this_cider) + "\n")
			out.write("########## meteor ##########\n")
			out.write(str(this_meteor) + "\n")
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


def gen_cider_input(file_in, file_out):
	'''
	generate json file for cider
	[{"image_id": "2008_006488.jpg", "caption": "Man in a boat fishing."}, 
	'''

	with open(file_in) as f:
		res = f.readlines()

	data = []

	for ind, this_line in enumerate(res):
		this_data = {}
		this_data["image_id"] = str(ind)
		this_data["caption"] = this_line.strip().decode('utf-8').encode('utf-8')
		data.append(this_data)

	with open(file_out, "w") as f:
		f.write(json.dumps(data))

	print ind



def merge_value_field_vocab(word_vocab_file, field_vocab_file, merge_vocab):
	'''
	merge word vocab with field vocab
	'''


	word_vocab = {}
	field_vocab = {}

	with open(word_vocab_file) as f:
		for line in f:
			line_list = line.strip().split("\t")
			word_vocab[line_list[0]] = int(line_list[1])

	with open(field_vocab_file) as f:
		for line in f:
			line_list = line.strip().split("\t")
			for each_word in line_list[0].split("_"):
				if each_word.strip() != "" and each_word not in word_vocab:
					word_vocab[each_word] = -1


	with open(merge_vocab, "w") as f:
		for tmp in word_vocab:
			f.write(tmp + "\t" + str(1) + "\n")

	print "All: ", len(word_vocab)


def add_vocab(origianl_vocab, add_vocab, out_vocab):
	'''
	extend vocab
	'''

	word_vocab = {}

	with open(origianl_vocab) as f:
		for line in f:
			line_list = line.strip().split("\t")
			if len(line_list) > 1:
				word_vocab[line_list[0]] = int(line_list[1])

	with open(add_vocab) as f:
		for line in f:
			line_list = line.strip().split("\t")
			if line_list[0] != "" and line_list[0] not in word_vocab:
				word_vocab[line_list[0]] = -1


	ind = 0
	with open(out_vocab, "w") as f:
		for tmp in word_vocab:
			f.write(tmp + "\t" + str(ind) + "\n")
			ind += 1

	print "All: ", len(word_vocab)




def create_ori_vocab(in_box, in_summary, word_vocab_file, field_vocab_file):
	'''
	create original vocab for each domain
	field: 2000
	word: 20000
	'''

	word_vocab = {}
	field_vocab = {}

	with open(in_box) as f:
		for line in f:
			for each_item in line.strip("\n").split("\t"):

				this_field_name = "_".join(each_item.split(":")[0].split("_")[:-1])
				if this_field_name not in field_vocab:
					field_vocab[this_field_name] = 0
				field_vocab[this_field_name] += 1

				this_value_token = each_item.split(":")[1]
				if this_value_token not in word_vocab:
					word_vocab[this_value_token] = 0
				word_vocab[this_value_token] += 1



	with open(in_summary) as f:
		for line in f:
			for token in line.strip().split(" "):
				if token not in word_vocab:
					word_vocab[token] = 0
				word_vocab[token] += 1


	sorted_word = sorted(word_vocab.items(), key=operator.itemgetter(1), reverse=True)
	sorted_field = sorted(field_vocab.items(), key=operator.itemgetter(1), reverse=True)

	print "All words: ", len(sorted_word)
	print "All fields: ", len(sorted_field)

	sorted_word = sorted_word[0:14999]


	with open(word_vocab_file, "w") as f:
		for item in sorted_word:
			f.write(item[0] + "\t" + str(item[1]) + "\n")

	with open(field_vocab_file, "w") as f:
		for item in sorted_field:
			f.write(item[0] + "\t" + str(item[1]) + "\n")


def extract_glove_vocab(file_in, file_out):
	'''
	extract vocab from glove embedding
	'''

	ind = 0
	f_out = open(file_out, "w")
	with open(file_in) as f:
		for line in f:
			line_list = line.strip().split()
			assert len(line_list) == 301

			f_out.write(line_list[0] + "\t" + str(ind) + "\n")
			ind += 1

	f_out.close()
	print ind




if __name__=='__main__':

	# file_in = summary_ours_in
	# file_out = "/scratch/home/zhiyu/wiki2bio/cider/data/candidate-baseline.json"
	# gen_cider_input(file_in, file_out)

	# file_in = summary_gold_in
	# file_out = "/scratch/home/zhiyu/wiki2bio/cider/data/references.json"
	# gen_cider_input(file_in, file_out)


	# gen_compare(cider_in, test_table, summary_ours_in, summary_with_unk, summary_gold_in, file_out)

	# metrics_dict = compute_metrics(hypothesis=summary_ours_in, references=[summary_gold_in])

	# print metrics_dict


	#check_field(train_table, "succession")

	#merge_value_field_vocab("/scratch/home/zhiyu/wiki2bio/original_data/")


	# in_box = "/scratch/home/zhiyu/wiki2bio/crawled_data/books.box"
	# in_summary = "/scratch/home/zhiyu/wiki2bio/crawled_data/books.summary"
	# word_vocab_file = "/scratch/home/zhiyu/wiki2bio/crawled_data/books_word_vocab.txt"
	# field_vocab_file = "/scratch/home/zhiyu/wiki2bio/crawled_data/books_field_vocab.txt"

	# # pc_all_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/personal_computers_word_vocab_all.txt"
	# # final_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/merged_vocab.txt"
	# # final_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/merged_field_vocab.txt"

	books_all_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/books_word_vocab_all.txt"
	songs_all_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/songs_word_vocab_all.txt"

	# books_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/books_field_vocab.txt"
	# songs_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/songs_field_vocab.txt"

	# create_ori_vocab(in_box, in_summary, word_vocab_file, field_vocab_file)
	# merge_value_field_vocab(word_vocab_file, field_vocab_file, books_all_vocab)

	ori_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/word_vocab.txt"
	# ori_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/merged_field_vocab.txt"

	final_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/pc_books_songs_word_vocab.txt"
	# final_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/pc_books_songs_field_vocab.txt"

	# add_vocab(ori_vocab, books_all_vocab, final_vocab)
	add_vocab(final_vocab, songs_all_vocab, final_vocab)


	# file_in = "/scratch/home/zhiyu/wiki2bio/other_data/glove.6B.300d.txt"
	# file_out = "/scratch/home/zhiyu/wiki2bio/emb_baseline/word_vocab.txt"
	# extract_glove_vocab(file_in, file_out)


	# file_in = "/scratch/home/zhiyu/wiki2bio/original_data/word_vocab.txt"
	# file_out = "/scratch/home/zhiyu/wiki2bio/crawled_data/word_vocab.txt"

	# vocab = {}
	# ind = 0
	# with open(file_in) as f:
	# 	for line in f:
	# 		line_list = line.strip().split()
	# 		vocab[line_list[0]] = int(line_list[1])
	# 		ind += 1

	# 		if ind > 14999:
	# 			break


	# print len(vocab)
	# with open(file_out, "w") as f:
	# 	for word in vocab:
	# 		f.write(word + "\t" + str(vocab[word]) + "\n")


	# ori_field = "/scratch/home/zhiyu/wiki2bio/original_data/field_vocab.txt"
	# file_in = "/scratch/home/zhiyu/wiki2bio/crawled_data/word_vocab_tmp.txt"
	# merge_value_field_vocab(file_in, ori_field, ori_vocab)



























