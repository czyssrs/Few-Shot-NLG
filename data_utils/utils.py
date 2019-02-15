import sys
import os
import operator
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords
import json
import zipfile
import string
import Queue
import random
import numpy as np
from nlgeval import NLGEval
from nlgeval import compute_metrics
from fuzzysearch import find_near_matches



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


def add_vocab_freq(origianl_vocab, add_vocab, out_vocab):
	'''
	extend vocab. exclude numbers
	'''

	word_vocab = {}

	with open(origianl_vocab) as f:
		for line in f:
			line_list = line.strip().split("\t")
			word = line_list[0]
			if not word.isdigit():
				if len(line_list) > 1:
					word_vocab[word] = int(line_list[1])

	with open(add_vocab) as f:
		for line in f:
			line_list = line.strip().split("\t")
			word = line_list[0]
			if not word.isdigit():
				if line_list[0] != "" and line_list[0] not in word_vocab:
					word_vocab[line_list[0]] = -1


	ind = 0
	with open(out_vocab, "w") as f:
		for tmp in word_vocab:
			f.write(tmp + "\t" + str(ind) + "\n")
			ind += 1

	print "All: ", len(word_vocab)




def create_ori_vocab(in_box, in_summary, word_vocab_file, field_vocab_file, size):
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
				if this_value_token.isdigit():
					continue
				if this_value_token not in word_vocab:
					word_vocab[this_value_token] = 0
				word_vocab[this_value_token] += 1



	with open(in_summary) as f:
		for line in f:
			for token in line.strip().split(" "):
				if token.isdigit():
					continue
				if token not in word_vocab:
					word_vocab[token] = 0
				word_vocab[token] += 1


	sorted_word = sorted(word_vocab.items(), key=operator.itemgetter(1), reverse=True)
	sorted_field = sorted(field_vocab.items(), key=operator.itemgetter(1), reverse=True)

	print "All words: ", len(sorted_word)
	print "All fields: ", len(sorted_field)

	sorted_word = sorted_word[0:size]


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

def read_word2vec_zip(word2vec_file):
    wordvec_map = {}
    num_words = 0
    dimension = 0
    zfile = zipfile.ZipFile(word2vec_file)
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        for line in ifile:
            line = line.strip()
            #print line
            entries = line.split(' ')
            if len(entries) == 2:
                continue
            word = entries[0].strip()
            vec = map(float, entries[1:])

            if word in wordvec_map:
                print ("Invalid word in embedding. Does not matter.")
                continue
            assert dimension == 0 or dimension == len(vec)

            wordvec_map[word] = np.array(vec)
            num_words += 1
            dimension = len(vec)

    return wordvec_map, num_words, dimension




def check_glove_coverage(glove_in, field_in):

	word2vec_map, _, _ = read_word2vec_zip(glove_in)

	word_vocab = {}
	with open(field_in) as f:
		for line in f:
			line_list = line.strip().split("\t")
			for each_word in line_list[0].split("_"):
				if each_word.strip() != "" and each_word not in word_vocab:
					word_vocab[each_word] = -1

	covered = 0
	for word in word_vocab:
		if word in word2vec_map:
			covered += 1


	print float(covered) / len(word_vocab)

def load_vocab(vocab_file):
	vocab = {}

	vocab['<_PAD>'] = 0
	vocab['<_START_TOKEN>'] = 1
	vocab['<_END_TOKEN>'] = 2
	vocab['<_UNK_TOKEN>'] = 3

	cnt = 4
	with open(vocab_file, "r") as v:
		for line in v:
			if len(line.strip().split()) > 1:
				word = line.strip().split()[0]
				ori_id = int(line.strip().split()[1])
				if word not in vocab:
					vocab[word] = (cnt + ori_id)

	return vocab

def load_local_vocab(vocab_file):
	vocab = {}

	cnt = 0
	with open(vocab_file, "r") as v:
		for line in v:
			if len(line.strip().split()) > 1:
				word = line.strip().split()[0]
				ori_id = int(line.strip().split()[1])
				if word not in vocab:
					vocab[word] = (cnt + ori_id)

	return vocab

def check_in_vocab(check_vocab, word_to_check):
	'''
	check if a word in glove vocab
	'''

	vocab = load_vocab(check_vocab)

	if word_to_check in vocab:
		print "Yes"
	else:
		print "No"


def process_songs(file_in, file_out):
	'''
	extra process remove ""
	`` 365 nichi kazoku '' is a single release by the japanese boyband kanjani8 .
	'''
	f_out = open(file_out, "w")
	with open(file_in) as f:
		for line in f:
			summary = line.strip()
			if summary[0] == "`":
				summary = summary[3:]
				sum_list = summary.split("''")
				if len(sum_list) > 1:
					summary = sum_list[0].strip() + "''".join(sum_list[1:])

			f_out.write(summary + "\n")

	f_out.close()


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


# def fuzzy_match(source, substring):

# 	res = find_near_matches(substring, source, max_deletions=3, max_insertions=3, max_substitutions=0)
# 	if len(res) == 0:
# 		return None

# 	result = res[0]

# 	fuzzy_res = source[result[0]:result[1]]
# 	if source[result[0] - 1] == " " and source[result[1]] == " ":

# 		### expand
# 		# forward
# 		before = source[:result[0]].strip().split(" ")[-1]
# 		if before in substring:
# 			fuzzy_res = before + " " + fuzzy_res
# 		after = source[result[1]:].strip().split(" ")[0]
# 		if after in substring:
# 			fuzzy_res = fuzzy_res + " " + after

# 		return fuzzy_res

# 	else:
# 		return None

def load_dem_map(file_in):
	'''
	recursively load nationality map
	'''
	dem_map = {}
	with open(file_in) as f:
		for line in f:
			line_list = line.strip().lower().split(",")
			if line_list[0] not in dem_map:
				dem_map[line_list[0]] = []
			if line_list[1] not in dem_map[line_list[0]]:
				dem_map[line_list[0]].append(line_list[1])

			if line_list[1] not in dem_map:
				dem_map[line_list[1]] = []
			if line_list[0] not in dem_map[line_list[1]]:
				dem_map[line_list[1]].append(line_list[0])

	final_res_map = {}
	for each_con in dem_map:
		res_con = []
		q = Queue.Queue()
		q.put(each_con)

		while not q.empty():
			con = q.get()
			if con in res_con:
				continue

			res_con.append(con)
			if con in dem_map:
				for each_sub in dem_map[con]:
					q.put(each_sub)

		final_res_map[each_con] = res_con

	return final_res_map


def fuzzy_match(source, substring, field_name):

	this_value = substring
	out_summary = source

	this_value_list_raw = this_value.split(" ")
	out_summary_list = out_summary.split(" ")
	# print this_value_list
	# print out_summary_list

	this_value_list = []
	for token in this_value_list_raw:
		if not(token in string.punctuation) \
			and token != "-lrb-" \
			and token != "-rrb-" \
			and token != "-lsb-" \
			and token != "-rsb-":
			this_value_list.append(token)

	if len(this_value_list) == 0:
		return out_summary

	num_consist = 0
	min_index = len(out_summary_list) + 1
	max_index = -1

	for token in this_value_list:
		if token in out_summary_list:
			num_consist += 1
			this_ind = out_summary_list.index(token)
			if this_ind < min_index:
				min_index = this_ind
			if this_ind > max_index:
				max_index = this_ind

	# print num_consist
	# print min_index
	# print max_index


	if float(num_consist) / len(this_value_list) > 0.4:
		if max_index - min_index <= 2 * len(this_value_list):
			### regard as match
			to_replace = " ".join(out_summary_list[min_index:max_index+1])
			if out_summary.startswith(to_replace):
				out_summary = out_summary.replace(to_replace + " ", "<" + field_name + "> ")
			else:
				out_summary = out_summary.replace(" " + to_replace + " ", " <" + field_name + "> ")

	return out_summary

def fuzzy_match_rep(source, substring, field_name):

	this_value = substring
	out_summary = source

	this_value_list_raw = this_value.split(" ")
	out_summary_list = out_summary.split(" ")
	# print this_value_list
	# print out_summary_list

	this_value_list = []
	for token in this_value_list_raw:
		if not(token in string.punctuation) \
			and token != "-lrb-" \
			and token != "-rrb-" \
			and token != "-lsb-" \
			and token != "-rsb-":
			this_value_list.append(token)

	if len(this_value_list) == 0:
		return out_summary

	num_consist = 0
	min_index = len(out_summary_list) + 1
	max_index = -1

	for token in this_value_list:
		if token in out_summary_list:
			num_consist += 1
			this_ind = out_summary_list.index(token)
			if this_ind < min_index:
				min_index = this_ind
			if this_ind > max_index:
				max_index = this_ind

	# print num_consist
	# print min_index
	# print max_index


	if float(num_consist) / len(this_value_list) > 0.4:
		if max_index - min_index <= 2 * len(this_value_list):
			### regard as match
			to_replace = " ".join(out_summary_list[min_index:max_index+1])
			replace_len = len(to_replace.split(" "))
			if out_summary.startswith(to_replace):
				out_summary = out_summary.replace(to_replace + " ", ("<" + field_name + "> ") * replace_len)
			else:
				out_summary = out_summary.replace(" " + to_replace + " ", " " + ("<" + field_name + "> ") * replace_len)

	return out_summary

def gen_mask(in_summary, in_box, out_summary, out_box, out_join):
	'''
	replace special token with unk
	'''

	### load nationality demonyms.csv
	dem_map = load_dem_map("/scratch/home/zhiyu/wiki2bio/other_data/demonyms.csv")


	with open(in_box) as f:
		lines_box = f.readlines()

	with open(in_summary) as f:
		lines_summary = f.readlines()

	out_s = open(out_summary, "w")
	out_b = open(out_box, "w")
	out_t = open(out_join, "w")

	for box, summary in zip (lines_box, lines_summary):

		box_list = box.strip().split("\t")
		box_out_list, box_field_list = join_box(box_list)

		out_summary = summary.strip()

		for (this_name, this_value) in box_field_list:


			if " " + this_value + " " in out_summary:

				out_summary = out_summary.replace(" " + this_value + " ", " <" + this_name + "> ")

			### name
			elif out_summary.startswith(this_value):
				out_summary = out_summary.replace(this_value, "<" + this_name + ">")

			### nationality
			elif this_value in dem_map:
				this_value_list = dem_map[this_value]
				for this_value in this_value_list:
					if " " + this_value + " " in out_summary:

						out_summary = out_summary.replace(" " + this_value + " ", " <" + this_name + "> ")


			else:

				## seperate nationality
				is_dem_match = 0
				this_value_list = this_value.split(" , ")
				if len(this_value_list) > 1:
					for each_con in this_value_list:
						if " " + each_con + " " in out_summary and each_con in dem_map:
							out_summary = out_summary.replace(" " + each_con + " ", " <" + this_name + "> ")
							is_dem_match = 1
							break
						if each_con in dem_map:
							this_con_list = dem_map[each_con]
							for this_con in this_con_list:
								if " " + this_con + " " in out_summary:
									out_summary = out_summary.replace(" " + this_con + " ", " <" + this_name + "> ")
									is_dem_match = 1
									break

				if is_dem_match:
					continue

				### fuzzy match 
				# match threshold? len percent? start - end index offset
				out_summary = fuzzy_match(out_summary, this_value, this_name)



		print box_list
		print box_field_list
		print out_summary
		print summary
		print "\n"

		out_b.write("\t".join([each_box[0] + ":" + each_box[1] for each_box in box_out_list]) + "\n")
		out_s.write(out_summary + "\n")

		out_t.write("\t".join([each_box[0] + ":" + each_box[1] for each_box in box_out_list]) + "\n")
		out_t.write(summary.strip() + "\n")
		out_t.write(out_summary + "\n")
		out_t.write("\n")



	out_s.close()
	out_b.close()
	out_t.close()



def gen_mask_field_pos(in_summary, in_box, out_field, out_pos, out_rpos, processed_summary, out_test):
	'''
	replace special token with unk
	'''

	### load nationality demonyms.csv
	dem_map = load_dem_map("/scratch/home/zhiyu/wiki2bio/other_data/demonyms.csv")

	data_path = "/scratch/home/zhiyu/wiki2bio/crawled_data/pointer/"
	sw = stopwords.words("english")
	freq_vocab = load_local_vocab(data_path + "human_books_songs_films_word_vocab_500.txt")


	with open(in_box) as f:
		lines_box = f.readlines()

	with open(in_summary) as f:
		lines_summary = f.readlines()

	with open(processed_summary) as f:
		lines_pro_summary = f.readlines()

	out_s = open(out_field, "w")
	out_p = open(out_pos, "w")
	out_rp = open(out_rpos, "w")

	out_t = open(out_test, "w")

	for box, summary, pro_summary in zip (lines_box, lines_summary, lines_pro_summary):

		box_list = box.strip().split("\t")
		box_out_list, box_field_list = join_box(box_list)

		tem_summary = summary.strip()
		out_summary = summary.strip()
		tem_summary_list = tem_summary.split(" ")

		out_field = np.zeros_like(out_summary.split(" ")).tolist()
		for ind in range(len(out_field)):
			out_field[ind] = '<_PAD>'

		out_pos, out_rpos = [], []

		for ind in range(len(out_field)):
			out_pos.append(0)
			out_rpos.append(0)

		for (this_name, this_value) in box_field_list:

			this_value_dict = {}
			for ind, each_token in enumerate(this_value.split(" ")):
				if each_token not in this_value_dict:
					this_value_dict[each_token] = ind + 1

			this_value_list_len = len(this_value.split(" "))

			if " " + this_value + " " in out_summary:

				out_summary = out_summary.replace(" " + this_value + " ", " " + ("<" + this_name + "> ") * this_value_list_len)



			### name
			elif out_summary.startswith(this_value + " "):
				out_summary = out_summary.replace(this_value + " ", ("<" + this_name + "> ") * this_value_list_len)

			### nationality
			elif this_value in dem_map:
				this_value_list = dem_map[this_value]
				for this_value in this_value_list:
					this_value_list_len = len(this_value.split(" "))
					if " " + this_value + " " in out_summary:

						out_summary = out_summary.replace(" " + this_value + " ", " " + ("<" + this_name + "> ") * this_value_list_len)


			else:

				## seperate nationality
				is_dem_match = 0
				this_value_list = this_value.split(" , ")
				if len(this_value_list) > 1:
					for each_con in this_value_list:
						if " " + each_con + " " in out_summary and each_con in dem_map:
							each_con_len = len(each_con.split(" "))
							out_summary = out_summary.replace(" " + each_con + " ", " " + ("<" + this_name + "> ") * each_con_len)
							is_dem_match = 1
							break
						if each_con in dem_map:
							this_con_list = dem_map[each_con]
							for this_con in this_con_list:
								if " " + this_con + " " in out_summary:
									this_con_len = len(this_con.split(" "))
									out_summary = out_summary.replace(" " + this_con + " ", " " + ("<" + this_name + "> ") * this_con_len)
									is_dem_match = 1
									break

				if is_dem_match:
					continue

				### fuzzy match 
				# match threshold? len percent? start - end index offset
				out_summary = fuzzy_match_rep(out_summary, this_value, this_name)

			assert len(out_summary.split(" ")) == len(tem_summary_list)

			for ind, token in enumerate(out_summary.split(" ")):
				if token == "<" + this_name + ">":
					out_field[ind] = this_name
					ori_token = tem_summary_list[ind]
					if ori_token in this_value_dict:
						out_pos[ind] = this_value_dict[ori_token]
						out_rpos[ind] = this_value_list_len - (out_pos[ind] - 1)

		# print box_list
		# print out_summary
		# print summary.strip()
		# print out_field
		# print out_pos
		# print out_rpos
		# print "\n"

		### second fuzzy match. by individual word
		for (this_name, this_value) in box_field_list:

			this_value_dict = {}
			for ind, each_token in enumerate(this_value.split(" ")):
				if each_token not in this_value_dict:
					this_value_dict[each_token] = ind + 1

			this_value_list_len = len(this_value.split(" "))
			this_value_list = this_value.split(" ")

			for ind, each_token in enumerate(out_summary.split(" ")):
				if (each_token not in sw) and each_token not in string.punctuation:
					if each_token in this_value_dict:
						out_summary.replace(" " + each_token + " ", " <" + this_name + "> ")

						out_field[ind] = this_name
						out_pos[ind] = this_value_dict[each_token]
						out_rpos[ind] = this_value_list_len - (out_pos[ind] - 1)


		assert len(out_summary.split(" ")) == len(tem_summary_list)





		print box_list
		print out_summary
		print summary.strip()
		print out_field
		print out_pos
		print out_rpos
		print "\n"

		# out_b.write("\t".join([each_box[0] + ":" + each_box[1] for each_box in box_out_list]) + "\n")
		# out_s.write(out_summary + "\n")

		out_t.write("\t".join([each_box[0] + ":" + each_box[1] for each_box in box_out_list]) + "\n")
		out_t.write("\n")
		out_t.write(summary.strip() + "\n")
		out_t.write("\n")
		out_t.write(out_summary + "\n")
		out_t.write("\n")
		out_t.write(pro_summary)
		out_t.write("\n")

		out_t.write("#########################\n")

		# print out_field
		# print len(out_field)
		# print tem_summary
		# print len(tem_summary_list)

		assert len(out_field) == len(tem_summary_list)
		assert len(tem_summary_list) == len(out_pos)
		assert len(tem_summary_list) == len(out_rpos)

		# for field_tmp, pos_tmp, rpos_tmp in zip(out_field, out_pos, out_rpos):
		# 	if field_tmp == "<_PAD>":
		# 		if pos_tmp != 0:
		# 			print box_list
		# 			print out_summary
		# 			print summary.strip()
		# 			print out_field
		# 			print out_pos
		# 			print out_rpos
		# 			print "\n"


		out_s.write(" ".join(out_field) + "\n")
		out_p.write(" ".join([str(tmp) for tmp in out_pos]) + "\n")
		out_rp.write(" ".join([str(tmp) for tmp in out_rpos]) + "\n")




	out_s.close()
	out_p.close()
	out_rp.close()


def dec_checker(summary_in, dec_in):

	with open(summary_in) as f_s:
		lines_s = f_s.readlines()

	with open(dec_in) as f_d:
		line_d = f_d.readlines()


	ind = 0
	for summary, dec in zip(lines_s, line_d):
		len_sum = len(summary.strip().split())
		len_dec = len(dec.strip().split())
		ind += 1


		if len_sum != len_dec:
			print str(len_sum) + "\t" + str(len_dec) + "\n"


def get_line_oov(file_in, file_out):
	'''
	utilitys.
	get number of lines with 3
	'''

	with open(file_in) as f:
		lines = f.readlines()

	without_3 = []

	for ind, line in enumerate(lines):

		# if "3" in line.strip().split():
		if "3" not in line.strip().split():
			without_3.append(ind)


	with open(file_out, "w") as f:
		f.write("\n".join([str(tmp) for tmp in without_3]))


def remove_oov(folder_in, folder_out, valid_line_in):


	with open(valid_line_in) as f:
		valid_line = [int(tmp.strip()) for tmp in f.readlines()]

	print len(valid_line)
	print "\n"


	for dirpath, dirnames, filenames in os.walk(folder_in):
		for filename in filenames:

			out_lines = []

			with open(folder_in + filename) as f:
				lines = [line.strip() for line in f.readlines()]


			for valid_ind in valid_line:
				out_lines.append(lines[valid_ind])

			print len(out_lines)

			with open(folder_out + filename, "w") as f_out:
				f_out.write("\n".join(out_lines))



def remove_oov_files(folder_in, folder_out, valid_line_in, filenames):


	with open(valid_line_in) as f:
		valid_line = [int(tmp.strip()) for tmp in f.readlines()]

	print len(valid_line)
	print "\n"


	# for dirpath, dirnames, filenames in os.walk(folder_in):
	for filename in filenames:

		out_lines = []

		with open(folder_in + filename) as f:
			lines = [line.strip() for line in f.readlines()]


		for valid_ind in valid_line:
			out_lines.append(lines[valid_ind])

		print len(out_lines)

		with open(folder_out + filename, "w") as f_out:
			f_out.write("\n".join(out_lines))


####### create description for each domain
def gen_sample_humans(in_box, in_summary, out_box, out_summary, sample_n):

	## human: name + birth_date, occupation, nationality, genre?
	# classic: name (born on ...) is a nationality occupation

	num_gen = 0
	num_birth = 0
	num_occupation = 0
	num_nationality = 0
	num_genre = 0

	with open(in_box) as f:
		box_lines = f.readlines()

	with open(in_summary) as f:
		summary_lines = f.readlines()

	out_b = open(out_box, "w")
	out_s = open(out_summary, "w")


	for line_box, line_summary in zip(box_lines, summary_lines):

		if num_gen >= sample_n:
			break

		box_list = line_box.strip().split("\t")
		box_out_list, box_field_list = join_box(box_list)

		this_box_dict = {}
		for (this_name, this_value) in box_field_list:
			this_box_dict[this_name] = this_value

		if "name" not in this_box_dict:
			continue

		this_name = this_box_dict["name"]

		field_iter = ["birth_date", "occupation", "occupation", "occupation", "nationality"]
		random.shuffle(field_iter)

		for field_target in field_iter:

			if field_target in this_box_dict:
				this_field_value = this_box_dict[field_target]

				###
				if field_target == "birth_date":
					num_birth += 1
					if num_birth > 35:
						continue
					dice = random.randint(1, 10)
					if dice < 7:
						this_out_summary = this_name + " is born on " + this_field_value + " ."
					else:
						this_out_summary = this_name + " , born " + this_field_value + " ."


				elif field_target == "occupation":
					num_occupation += 1
					if num_occupation > 35:
						continue
					this_out_summary = this_name + " is a " + this_field_value + " ."

				elif field_target == "nationality":
					num_nationality += 1
					if num_nationality > 35:
						continue
					dice = random.randint(1, 10)
					if dice < 5:
						this_out_summary = this_name + " is from " + this_field_value + " ."
					else:
						this_out_summary = this_name + " is a " + this_field_value + " ."


				# ### ?
				# elif field_target == "genre":
				# 	dice = random.randint(1, 10)
				# 	if dice < 5:
				# 		this_out_summary = this_name + " is from " + this_field_value + " ."
				# 	else:
				# 		this_out_summary = this_name + " is a " + this_field_value + " ."


				this_out_box = []

				if len(this_name.split(" ")) < 2:
					this_out_box.append("name:" + this_name)
				else:
					for ind, token in enumerate(this_name.split(" ")):
						this_out_box.append("name_" + str(ind + 1) + ":" + token)

				if len(this_field_value.split(" ")) < 2:
					this_out_box.append(field_target + ":" + this_field_value)
				else:
					for ind, token in enumerate(this_field_value.split(" ")):
						this_out_box.append(field_target + "_" + str(ind + 1) + ":" + token)


				out_b.write("\t".join(this_out_box) + "\n")
				out_s.write(this_out_summary + "\n")

				print this_out_box
				print this_out_summary

				num_gen += 1

				break




	out_b.close()
	out_s.close()

def gen_sample_books(in_box, in_summary, out_box, out_summary, sample_n):

	## human: name + author, genre, publisher, publication_date, country
	# classic: name is a country genre novel by author, published by publisher on publication_date

	num_gen = 0
	num_1 = 0
	num_2 = 0
	num_3 = 0
	num_4 = 0
	num_5 = 0

	with open(in_box) as f:
		box_lines = f.readlines()

	with open(in_summary) as f:
		summary_lines = f.readlines()

	out_b = open(out_box, "w")
	out_s = open(out_summary, "w")


	for line_box, line_summary in zip(box_lines, summary_lines):

		if num_gen >= sample_n:
			break

		box_list = line_box.strip().split("\t")
		box_out_list, box_field_list = join_box(box_list)

		this_box_dict = {}
		for (this_name, this_value) in box_field_list:
			this_box_dict[this_name] = this_value

		if "name" not in this_box_dict:
			continue

		this_name = this_box_dict["name"]

		field_iter = ["author", "genre", "publisher", "publication_date", "country"]
		random.shuffle(field_iter)

		for field_target in field_iter:

			if field_target in this_box_dict:
				this_field_value = this_box_dict[field_target]

				###
				if field_target == "author":
					if num_1 > 20:
						continue
					num_1 += 1
					dice = random.randint(1, 10)
					if dice < 6:
						this_out_summary = this_name + " is a novel by " + this_field_value + " ."
					elif dice < 8:
						this_out_summary = this_name + " is a novel written by " + this_field_value + " ."
					elif dice < 10:
						this_out_summary = this_name + " is written by " + this_field_value + " ."
					else:
						this_out_summary = this_name + " is a novel authored by " + this_field_value + " ."


				elif field_target == "genre":
					if this_field_value.split()[-1] != "novel":
						if num_2 > 20:
							continue
						num_2 += 1
						this_out_summary = this_name + " is a " + this_field_value + " novel ."
					else:
						continue

				elif field_target == "publisher":
					if num_3 > 20:
						continue
					num_3 += 1
					dice = random.randint(1, 10)
					if dice < 5:
						this_out_summary = this_name + " is published by " + this_field_value + " ."
					else:
						this_out_summary = this_name + " is a novel published by " + this_field_value + " ."

				elif field_target == "publication_date":
					if num_4 > 20:
						continue
					num_4 += 1
					dice = random.randint(1, 10)
					if dice < 7:
						this_out_summary = this_name + " is a novel published in " + this_field_value + " ."
					else:
						this_out_summary = this_name + " is a " + this_field_value + " novel ."

				elif field_target == "country":
					if num_5 > 20:
						continue
					num_5 += 1
					this_out_summary = this_name + " is a " + this_field_value + " novel ."


				this_out_box = []

				if len(this_name.split(" ")) < 2:
					this_out_box.append("name:" + this_name)
				else:
					for ind, token in enumerate(this_name.split(" ")):
						this_out_box.append("name_" + str(ind + 1) + ":" + token)

				if len(this_field_value.split(" ")) < 2:
					this_out_box.append(field_target + ":" + this_field_value)
				else:
					for ind, token in enumerate(this_field_value.split(" ")):
						this_out_box.append(field_target + "_" + str(ind + 1) + ":" + token)


				out_b.write("\t".join(this_out_box) + "\n")
				out_s.write(this_out_summary + "\n")

				print this_out_box
				print this_out_summary

				num_gen += 1

				break




	print num_1
	print num_2
	print num_3
	print num_4
	print num_5

	out_b.close()
	out_s.close()

### TODO: gen sample for songs, films


if __name__=='__main__':


	### generate domain examples
	in_box = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/original_data_oov/books.box"
	in_summary = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/original_data_oov/books.summary"
	out_box = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/domain_descriptions/books.box"
	out_summary = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/domain_descriptions/books.summary"

	gen_sample_books(in_box, in_summary, out_box, out_summary, 100)

	# ## remove 3 in all original processed files
	# file_in = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/processed_data_withoov/test/test.summary.id"
	# file_out = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/books_lines_invalid.txt"
	# get_line_oov(file_in, file_out)

	# folder_in = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/processed_data_withoov/test/"
	# folder_out = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/processed_data/test/"

	# oov_line_in = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/test_lines_valid.txt"

	# remove_oov(folder_in, folder_out, oov_line_in)



	# folder_in = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/original_data_withoov/"
	# folder_out = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/original_data_oov/"

	# valid_line_in = "/scratch/home/zhiyu/wiki2bio/emb_pointer_copyloss_am/humans_lines_invalid.txt"
	# file_names = ["valid.summary", "valid.box"]


	# remove_oov_files(folder_in, folder_out, valid_line_in, file_names)




	# ### generate mask
	# in_summary = "/scratch/home/zhiyu/wiki2bio/original_data/valid.summary"
	# in_box = "/scratch/home/zhiyu/wiki2bio/original_data/valid.box"
	# out_field = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/valid.field"
	# out_pos = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/valid.pos"
	# out_rpos = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/valid.rpos"

	# processed_summary = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/processed_data/valid/valid.summary.id"
	# test_case = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/original_valid_case.txt"

	# gen_mask_field_pos(in_summary, in_box, out_field, out_pos, out_rpos, processed_summary, test_case)


	# dec_in = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/processed_data/train/train.summary.id"
	# summary_in = "/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/original_data/train.summary"
	# dec_checker(summary_in, dec_in)

	# herbert <article_title> , jr. -lrb- born april 21 , 1945 in <birth_place> -rrb- is a former american football player .

	# out_summary = "hassan taftian -lrb- ; born 4 may 1993 in torbat-e heydarieh -rrb- is an iranian sprinter ."
	# this_value = "04 may 1993"
	# print fuzzy_match(source, substring, "replace")




	# in_summary = "/scratch/home/zhiyu/wiki2bio/original_data/test.summary"
	# in_box = "/scratch/home/zhiyu/wiki2bio/original_data/test.box"
	# out_file = "/scratch/home/zhiyu/wiki2bio/masked_summary.txt"
	# gen_mask(in_summary, in_box, out_file)

	# file_in = "/scratch/home/zhiyu/wiki2bio/crawled_data/songs.summary.bak"
	# file_out = "/scratch/home/zhiyu/wiki2bio/crawled_data/songs.summary"
	# process_songs(file_in, file_out)

	# ori_vocab = "/scratch/home/zhiyu/wiki2bio/original_data/field_vocab.txt"

	# check_in_vocab(ori_vocab, "published")

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


	# data_path = "/scratch/home/zhiyu/wiki2bio/crawled_data/pointer/"

	# # domain = "books"
	# # in_box = data_path + domain + ".box"
	# # in_summary = data_path + domain + ".summary"
	# # word_vocab_file = data_path + domain + "_word_vocab.txt"
	# # field_vocab_file = data_path + domain + "_field_vocab.txt"

	# # # # pc_all_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/personal_computers_word_vocab_all.txt"
	# # # # final_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/merged_vocab.txt"
	# # # # final_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/merged_field_vocab.txt"

	# books_all_vocab = data_path + "books_word_vocab_all.txt"
	# songs_all_vocab = data_path + "songs_word_vocab_all.txt"
	# films_all_vocab = data_path + "films_word_vocab_all.txt"

	# # # # books_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/books_field_vocab.txt"
	# # # # songs_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/songs_field_vocab.txt"
	# # # # films_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/films_field_vocab.txt"

	# # create_ori_vocab(in_box, in_summary, word_vocab_file, field_vocab_file, 1999)
	# # merge_value_field_vocab(word_vocab_file, field_vocab_file, books_all_vocab)

	# ori_vocab = data_path + "word_vocab_2000.txt"
	# final_vocab = data_path + "human_books_songs_films_word_vocab_2000.txt"

	# # # # ori_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/field_vocab.txt"
	# # # # final_field_vocab = "/scratch/home/zhiyu/wiki2bio/crawled_data/human_books_songs_films_field_vocab.txt"

	# add_vocab(ori_vocab, books_all_vocab, final_vocab)
	# add_vocab(final_vocab, songs_all_vocab, final_vocab)
	# add_vocab(final_vocab, films_all_vocab, final_vocab)

	# add_vocab(ori_field_vocab, books_field_vocab, final_field_vocab)
	# add_vocab(final_field_vocab, songs_field_vocab, final_field_vocab)
	# add_vocab(final_field_vocab, films_field_vocab, final_field_vocab)


	# ori_vocab = data_path + "word_vocab_200.txt"

	# books_all_vocab = data_path + "books_word_vocab_200.txt"
	# songs_all_vocab = data_path + "songs_word_vocab_200.txt"
	# films_all_vocab = data_path + "films_word_vocab_200.txt"

	# final_vocab = data_path + "human_books_songs_films_word_vocab_200.txt"

	# add_vocab_freq(ori_vocab, books_all_vocab, final_vocab)
	# add_vocab_freq(final_vocab, songs_all_vocab, final_vocab)
	# add_vocab_freq(final_vocab, films_all_vocab, final_vocab)


	# file_in = "/scratch/home/zhiyu/wiki2bio/other_data/glove.6B.300d.txt"
	# file_out = "/scratch/home/zhiyu/wiki2bio/emb_baseline/word_vocab.txt"
	# extract_glove_vocab(file_in, file_out)


	# file_in = "/scratch/home/zhiyu/wiki2bio/original_data/word_vocab.txt"
	# file_out = "/scratch/home/zhiyu/wiki2bio/crawled_data/pointer/word_vocab_2000.txt"

	# vocab = []
	# ind = 0
	# with open(file_in) as f:
	# 	for line in f:
	# 		line_list = line.strip().split()
	# 		if line_list[0].isdigit():
	# 			continue
	# 		vocab.append((line_list[0], line_list[1]))
	# 		ind += 1

	# 		if ind > 1999:
	# 			break


	# print len(vocab)
	# with open(file_out, "w") as f:
	# 	for word in vocab:
	# 		f.write(word[0] + "\t" + word[1] + "\n")


	# ori_field = "/scratch/home/zhiyu/wiki2bio/original_data/field_vocab.txt"
	# file_in = "/scratch/home/zhiyu/wiki2bio/crawled_data/word_vocab_tmp.txt"
	# merge_value_field_vocab(file_in, ori_field, ori_vocab)


	# glove_in = "/scratch/home/zhiyu/wiki2bio/other_data/glove.42B.300d.zip"
	# field_in = "/scratch/home/zhiyu/wiki2bio/crawled_data/books_field_vocab.txt"
	# check_glove_coverage(glove_in, field_in)



























