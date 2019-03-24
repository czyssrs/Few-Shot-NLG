import re, time, os, sys
import json
import zipfile
import string
import queue
import encoder
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords

# root_path = "/scratch/home/zhiyu/wiki2bio/"
root_path = "../few_shot_gpt-2/"
# merge_word_vocab = "crawled_data/merged_vocab.txt"

merge_field_vocab = root_path + "human_books_songs_films_field_vocab.txt"

### bpe vocab
enc = encoder.get_encoder("117M")
# "empty": 28920
# field_empty = 28920
field_empty = 5713


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

    # random_out = random.shuffle(sorted_by_second)

    return out_list, sorted_by_second

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
        q = queue.Queue()
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
            and token != "(" \
            and token != ")" \
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

def gen_mask_field_pos(in_summary, in_box, out_field, out_pos, out_rpos):
    '''
    replace special token with unk
    '''

    ### load nationality demonyms.csv
    dem_map = load_dem_map("/scratch/home/zhiyu/wiki2bio/other_data/demonyms.csv")

    # sw = stopwords.words("english")
    # freq_vocab = load_local_vocab(root_path + "human_books_songs_films_word_vocab_200.txt")


    with open(in_box) as f:
        lines_box = f.readlines()

    with open(in_summary) as f:
        lines_summary = f.readlines()

    out_s = open(out_field, "w")
    out_p = open(out_pos, "w")
    out_rp = open(out_rpos, "w")

    for box, summary in tqdm(zip(lines_box, lines_summary)):


        box = box.replace("-lrb-", "(")
        box = box.replace("-rrb-", ")")

        box_list = box.strip().split("\t")
        box_out_list, box_field_list = join_box(box_list)

        summary = summary.replace("-lrb-", "(")
        summary = summary.replace("-rrb-", ")")

        tem_summary = summary.strip()
        out_summary = summary.strip()
        tem_summary_list = tem_summary.split(" ")


        out_pos, out_rpos, out_field = [], [], []
        out_pos_bpe, out_rpos_bpe, out_field_bpe = [], [], []

        out_bpe, _ = enc.encode(summary.strip())
        out_bpe_len = len(out_bpe)

        for ind in range(out_bpe_len):
            out_pos_bpe.append(0)
            out_rpos_bpe.append(0)

        for ind in range(out_bpe_len):
            out_field_bpe.append('empty')

        for ind in range(len(tem_summary_list)):
            out_pos.append(0)
            out_rpos.append(0)

        for ind in range(len(tem_summary_list)):
            out_field.append('empty')


        for (this_name, this_value) in box_field_list:

            this_value_dict = {}
            this_pos_bpe_dict = {}
            prev = 1
            for ind, each_token in enumerate(this_value.split(" ")):
                # if each_token not in this_value_dict:
                this_value_dict[each_token] = ind + 1



                if this_name != "name":
                    each_token = " " + each_token
                else:
                    if ind != 0:
                        each_token = " " + each_token

                bpe_tokens, bpe_tokens_original = enc.encode(each_token)

                ### (start ind, len)
                this_pos_bpe_dict[ind + 1] = (prev, len(bpe_tokens))
                prev += len(bpe_tokens)


            if this_name == "name":
                bpe_value = this_value
            else:
                bpe_value = " " + this_value
            bpe_tokens, bpe_tokens_original = enc.encode(bpe_value)

            this_value_bpe_len = len(bpe_tokens)
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


                    ### convert to bpe
                    ori_token_bpe = ori_token
                    if ind != 0:
                        ori_token_bpe = " " + ori_token

                    if ind > 0:
                        past = tem_summary_list[:ind]
                        past = " ".join(past)
                        bpe_past, _ = enc.encode(past)
                        past_len = len(bpe_past)

                    else:

                        past_len = 0



                    bpe_tokens, bpe_tokens_original = enc.encode(ori_token_bpe)
                    for it in range(len(bpe_tokens)):
                        out_field_bpe[past_len + it] = this_name


                    if ori_token in this_value_dict:
                        bpe_pos_start, bpe_pos_len = this_pos_bpe_dict[out_pos[ind]]
                        for it in range(bpe_pos_len):
                            start = bpe_pos_start + it
                            end = this_value_bpe_len - (start - 1)
                            if start > 30:
                                start = 30
                            if end > 30:
                                end = 30
                            if past_len + it >= len(out_pos_bpe):
                                this_id = past_len
                            else:
                                this_id = past_len + it
                            out_pos_bpe[this_id] = start
                            out_rpos_bpe[this_id] = end



        # ### bpe for debug
        # box_out_list_bpe = {}
        # for (name, value) in box_out_list:
        #     if name != "name":
        #         value = " " + value
        #     bpe_tokens, bpe_tokens_original = enc.encode(value)
        #     box_out_list_bpe[name] = " ".join(bpe_tokens_original)

        bpe_tokens, bpe_tokens_original = enc.encode(summary.strip())
        bpe_test = " ".join(bpe_tokens_original)


        # print (box_out_list)
        # print ("######################################")
        # print (summary.strip())
        # print ("######################################")
        # print (out_field)
        # print ("######################################")
        # print (box_out_list_bpe)
        # print ("######################################")
        # print (bpe_test)
        # print ("######################################")
        # print (out_field_bpe)
        # print ("######################################")
        # print (out_pos_bpe)
        # print ("######################################")
        # print (out_rpos_bpe)
        # print ("\n")

        # ### second fuzzy match. by individual word
        # for (this_name, this_value) in box_field_list:

        #     this_value_dict = {}
        #     for ind, each_token in enumerate(this_value.split(" ")):
        #         if each_token not in this_value_dict:
        #             this_value_dict[each_token] = ind + 1

        #     this_value_list_len = len(this_value.split(" "))
        #     this_value_list = this_value.split(" ")

        #     for ind, each_token in enumerate(out_summary.split(" ")):

        #         ### most freq 500 still looks domain specific
        #         if (each_token not in sw) and (each_token not in freq_vocab):
        #         # if (each_token not in sw):
        #             if each_token in this_value_dict:
        #                 out_summary.replace(" " + each_token + " ", " <" + this_name + "> ")

        #                 out_field[ind] = this_name
        #                 out_pos[ind] = this_value_dict[each_token]
        #                 out_rpos[ind] = this_value_list_len - (out_pos[ind] - 1)


        assert len(out_summary.split(" ")) == len(tem_summary_list)


        assert len(out_field) == len(tem_summary_list)
        assert len(tem_summary_list) == len(out_pos)
        assert len(tem_summary_list) == len(out_rpos)

        assert len(out_field_bpe) == len(bpe_tokens)
        assert len(out_pos_bpe) == len(bpe_tokens)
        assert len(out_rpos_bpe) == len(bpe_tokens)

        # for field_tmp, pos_tmp, rpos_tmp in zip(out_field, out_pos, out_rpos):
        #   if field_tmp == "<_PAD>":
        #       if pos_tmp != 0:
        #           print box_list
        #           print out_summary
        #           print summary.strip()
        #           print out_field
        #           print out_pos
        #           print out_rpos
        #           print "\n"

        ### convert to bpe




        out_s.write(" ".join(out_field_bpe) + "\n")
        out_p.write(" ".join([str(tmp) for tmp in out_pos_bpe]) + "\n")
        out_rp.write(" ".join([str(tmp) for tmp in out_rpos_bpe]) + "\n")



    out_s.close()
    out_p.close()
    out_rp.close()


def gen_context(domain):
    '''
    trail: generate context for gpt
    '''
    boxes = [root_path + domain + "/original_data/train.box", root_path + domain + "/original_data/valid.box", root_path + domain + "/original_data/test.box"]
    context = [root_path + domain + "/processed_data/train/train.context", 
                root_path + domain + "/processed_data/valid/valid.context", 
                root_path + domain + "/processed_data/test/test.context"]
    
    mixb_word, mixb_label, mixb_pos = [], [], []
    for fboxes in boxes:
        box = open(fboxes, "r").read().strip().split('\n')
        box_word, box_label, box_pos = [], [], []
        for ib in box:

            ib = ib.replace("-lrb-", "(")
            ib = ib.replace("-rrb-", ")")

            box_single_word, box_single_label, box_single_pos = [], [], []
            item = ib.split('\t')

            box_out_list, _ = join_box(item)

            for (this_name, this_value) in box_out_list:

                if '<none>' in this_value:
                    continue

                this_value = " " + this_value

                if this_name != "name":
                    context_name = this_name.replace("_", " ")

                # this_value = this_value.replace("-lrb-", "(")
                # this_value = this_value.replace("-rrb-", ")")

                tokens, tokens_original = enc.encode(this_value)

                for ind, each_token in enumerate(tokens_original):
                    box_single_word.append(each_token)
                    box_single_label.append(this_name)
                    box_single_pos.append(ind + 1  if ind + 1<=30 else 30)






def split_infobox(domain):
    """
    extract box content, field type and position information from infoboxes from original_data
    *.box.val is the box content (token)
    *.box.lab is the field type for each token
    *.box.pos is the position counted from the begining of a field
    """
    bwfile = [root_path + domain + "/processed_data/train/train.box.val", 
              root_path + domain + "/processed_data/valid/valid.box.val", 
              root_path + domain + "/processed_data/test/test.box.val"]
    bffile = [root_path + domain + "/processed_data/train/train.box.lab", 
              root_path + domain + "/processed_data/valid/valid.box.lab", 
              root_path + domain + "/processed_data/test/test.box.lab"]
    bpfile = [root_path + domain + "/processed_data/train/train.box.pos", 
              root_path + domain + "/processed_data/valid/valid.box.pos", 
              root_path + domain + "/processed_data/test/test.box.pos"]
    boxes = [root_path + domain + "/original_data/train.box", root_path + domain + "/original_data/valid.box", root_path + domain + "/original_data/test.box"]
    
    mixb_word, mixb_label, mixb_pos = [], [], []
    for fboxes in boxes:
        box = open(fboxes, "r").read().strip().split('\n')
        box_word, box_label, box_pos = [], [], []
        for ib in box:

            ib = ib.replace("-lrb-", "(")
            ib = ib.replace("-rrb-", ")")

            box_single_word, box_single_label, box_single_pos = [], [], []
            item = ib.split('\t')

            box_out_list, _ = join_box(item)

            for (this_name, this_value) in box_out_list:

                if '<none>' in this_value:
                    continue

                if this_name != "name":
                    this_value = " " + this_value

                # this_value = this_value.replace("-lrb-", "(")
                # this_value = this_value.replace("-rrb-", ")")

                tokens, tokens_original = enc.encode(this_value)

                for ind, each_token in enumerate(tokens_original):
                    box_single_word.append(each_token)
                    box_single_label.append(this_name)
                    box_single_pos.append(ind + 1  if ind + 1<=30 else 30)


            # print (box_out_list)
            # print (box_single_word)
            # print (box_single_label)
            # print (box_single_pos)
            # print ("\n")


            # for it in box_out_list:
            #     if len(it.split(':')) > 2:
            #         continue
            #     # print it
            #     prefix, word = it.split(':')
            #     if '<none>' in word or word.strip()=='' or prefix.strip()=='':
            #         continue
            #     new_label = re.sub("_[1-9]\d*$", "", prefix)
            #     if new_label.strip() == "":
            #         continue
            #     box_single_word.append(word)
            #     box_single_label.append(new_label)
            #     if re.search("_[1-9]\d*$", prefix):
            #         field_id = int(prefix.split('_')[-1])
            #         box_single_pos.append(field_id if field_id<=30 else 30)
            #     else:
            #         box_single_pos.append(1)


            box_word.append(box_single_word)
            box_label.append(box_single_label)
            box_pos.append(box_single_pos)
        mixb_word.append(box_word)
        mixb_label.append(box_label)
        mixb_pos.append(box_pos)
    for k, m in enumerate(mixb_word):
        with open(bwfile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_label):
        with open(bffile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')
    for k, m in enumerate(mixb_pos):
        with open(bpfile[k], "w+") as h:
            for items in m:
                for sens in items:
                    h.write(str(sens) + " ")
                h.write('\n')


def reverse_pos(domain):
    # get the position counted from the end of a field
    bpfile = [root_path + domain + "/processed_data/train/train.box.pos", \
        root_path + domain + "/processed_data/valid/valid.box.pos", root_path + domain + "/processed_data/test/test.box.pos"]

    bwfile = [root_path + domain + "/processed_data/train/train.box.rpos", \
        root_path + domain + "/processed_data/valid/valid.box.rpos", root_path + domain + "/processed_data/test/test.box.rpos"]
    for k, pos in enumerate(bpfile):
        box = open(pos, "r").read().strip().split('\n')
        reverse_pos = []
        for bb in box:
            pos = bb.split()
            tmp_pos = []
            single_pos = []
            for p in pos:
                if int(p) == 1 and len(tmp_pos) != 0:
                    single_pos.extend(tmp_pos[::-1])
                    tmp_pos = []
                tmp_pos.append(p)
            single_pos.extend(tmp_pos[::-1])
            reverse_pos.append(single_pos)
        with open(bwfile[k], 'w+') as bw:
            for item in reverse_pos:
                bw.write(" ".join(item) + '\n')

def check_generated_box(domain):
    ftrain = [root_path + domain + "/processed_data/train/train.box.val",
              root_path + domain + "/processed_data/train/train.box.lab",
              root_path + domain + "/processed_data/train/train.box.pos",
              root_path + domain + "/processed_data/train/train.box.rpos"]
    ftest  = [root_path + domain + "/processed_data/test/test.box.val", 
              root_path + domain + "/processed_data/test/test.box.lab",
              root_path + domain + "/processed_data/test/test.box.pos",
              root_path + domain + "/processed_data/test/test.box.rpos"]
    fvalid = [root_path + domain + "/processed_data/valid/valid.box.val", 
              root_path + domain + "/processed_data/valid/valid.box.lab", 
              root_path + domain + "/processed_data/valid/valid.box.pos",
              root_path + domain + "/processed_data/valid/valid.box.rpos"]
    for case in [ftrain, ftest, fvalid]:
        vals = open(case[0], 'r').read().strip().split('\n')
        labs = open(case[1], 'r').read().strip().split('\n')
        poses = open(case[2], 'r').read().strip().split('\n')
        rposes = open(case[3], 'r').read().strip().split('\n')
        assert len(vals) == len(labs)
        assert len(poses) == len(labs)
        assert len(rposes) == len(poses)
        for val, lab, pos, rpos in zip(vals, labs, poses, rposes):
            vval = val.strip().split(' ')
            llab = lab.strip().split(' ')
            ppos = pos.strip().split(' ')
            rrpos = rpos.strip().split(' ')
            if len(vval) != len(llab) or len(llab) != len(ppos) or len(ppos) != len(rrpos):
                print (case)
                print (val)
                print (len(vval))
                print (len(llab))
                print (len(ppos))
                print (len(rrpos))
            assert len(vval) == len(llab)
            assert len(llab) == len(ppos)
            assert len(ppos) == len(rrpos)


def split_summary_for_rouge(domain):
    bpfile = [root_path + domain + "/original_data/test.summary", root_path + domain + "/original_data/valid.summary"]
    bwfile = [root_path + domain + "/processed_data/test/test_split_for_rouge/", root_path + domain + "/processed_data/valid/valid_split_for_rouge/"]
    for i, fi in enumerate(bpfile):
        fread = open(fi, 'r')
        k = 0
        for line in fread:
            with open(bwfile[i] + 'gold_summary_' + str(k), 'w') as sw:
                sw.write(line.strip() + '\n')
            k += 1
        fread.close()



def table2id(domain):
    fvals = [root_path + domain + '/processed_data/train/train.box.val',
             root_path + domain + '/processed_data/test/test.box.val',
             root_path + domain + '/processed_data/valid/valid.box.val']
    flabs = [root_path + domain + '/processed_data/train/train.box.lab',
             root_path + domain + '/processed_data/test/test.box.lab',
             root_path + domain + '/processed_data/valid/valid.box.lab']
    fsums = [root_path + domain + '/original_data/train.summary',
             root_path + domain + '/original_data/test.summary',
             root_path + domain + '/original_data/valid.summary']

    fvals2id = [root_path + domain + '/processed_data/train/train.box.val.id',
                root_path + domain + '/processed_data/test/test.box.val.id',
                root_path + domain + '/processed_data/valid/valid.box.val.id']
    flabs2id = [root_path + domain + '/processed_data/train/train.box.lab.id',
                root_path + domain + '/processed_data/test/test.box.lab.id',
                root_path + domain + '/processed_data/valid/valid.box.lab.id']
    fsums2id = [root_path + domain + '/processed_data/train/train.summary.id',
                root_path + domain + '/processed_data/test/test.summary.id',
                root_path + domain + '/processed_data/valid/valid.summary.id']



    f_local_vocab = [root_path + domain + '/processed_data/train/train_local_oov.txt',
                    root_path + domain + '/processed_data/test/test_local_oov.txt',
                    root_path + domain + '/processed_data/valid/valid_local_oov.txt']

    f_decoder_field = [root_path + domain + '/processed_data/train/train_summary_field.txt',
                        root_path + domain + '/processed_data/test/test_summary_field.txt',
                        root_path + domain + '/processed_data/valid/valid_summary_field.txt']

    f_decoder_field_id = [root_path + domain + '/processed_data/train/train_summary_field_id.txt',
                        root_path + domain + '/processed_data/test/test_summary_field_id.txt',
                        root_path + domain + '/processed_data/valid/valid_summary_field_id.txt']

    f_decoder_pos = [root_path + domain + '/processed_data/train/train_summary_pos.txt',
                    root_path + domain + '/processed_data/test/test_summary_pos.txt',
                    root_path + domain + '/processed_data/valid/valid_summary_pos.txt']

    f_decoder_rpos = [root_path + domain + '/processed_data/train/train_summary_rpos.txt',
                    root_path + domain + '/processed_data/test/test_summary_rpos.txt',
                    root_path + domain + '/processed_data/valid/valid_summary_rpos.txt']

    boxes = [root_path + domain + "/original_data/train.box", root_path + domain + "/original_data/test.box", root_path + domain + "/original_data/valid.box"]



    ### write field to word mapping
    key_map = dict()
    key_map['empty'] = 0
    cnt = 1
    with open(merge_field_vocab, "r") as v:
        for line in v:
            key = line.strip().split()[0]
            key_map[key] = cnt
            cnt += 1
    key2id = key_map
    id2key = {value: key for key, value in key_map.items()}
    print (len(key_map))

    #### add for field id to word group mapping
    keyid2wordlist =  dict()


    for i in range(0, len(id2key)):

        if i == 0:
            bpe_in = id2key[i].replace("_", " ")
        else:
            bpe_in = " " + id2key[i].replace("_", " ")
        bpe_tokens, bpe_token_original = enc.encode(bpe_in)
        keyid2wordlist[i] = bpe_tokens

        if len(keyid2wordlist[i]) > 3:
            keyid2wordlist[i] = keyid2wordlist[i][:3]
        else:
            extended = 3 - len(keyid2wordlist[i])
            keyid2wordlist[i] += ([field_empty] * extended)

    field2word_file = root_path + domain + "/processed_data/field2word.txt"
    with open(field2word_file, "w") as f:
        for each_id in keyid2wordlist:
            f.write(str(each_id) + "\t" + " ".join([str(tmp) for tmp in keyid2wordlist[each_id]]) + "\n")


    ### field not change
    for k, ff in enumerate(flabs):
        fi = open(ff, 'r')
        fo = open(flabs2id[k], 'w')
        for line in fi:
            items = line.strip().split()
            # print (items)
            res_items = []
            for key in items:
                if key in key2id:
                    res_items.append(str(key2id[key]))
                else:
                    res_items.append("0")

            fo.write(" ".join(res_items) + '\n')
        fi.close()
        fo.close()





    ### gen field, pos for decoder. bpe

    for k, (fs, fb) in enumerate(zip(fsums, boxes)):

        gen_mask_field_pos(fs, fb, f_decoder_field[k], f_decoder_pos[k], f_decoder_rpos[k])


    for k, ff in enumerate(f_decoder_field):
        fi = open(ff, 'r')
        fo = open(f_decoder_field_id[k], 'w')
        for line in fi:
            items = line.strip().split()
            res_items = []
            for key in items:
                if key in key2id:
                    res_items.append(str(key2id[key]))
                else:
                    res_items.append("0")
            fo.write(" ".join(res_items) + '\n')
        fi.close()
        fo.close()


    for k, (fs, fv) in enumerate(zip(fsums, fvals)):
        fsum = open(fs)
        fsumo = open(fsums2id[k], 'w')

        fval = open(fv)
        fvalo = open(fvals2id[k], 'w')

        lines_sum = fsum.readlines()
        lines_val = fval.readlines()


        for line_sum, line_val in zip(lines_sum, lines_val):

            line_val_list = line_val.strip().split()
            res_val_list = []
            for bpe_token in line_val_list:
                if bpe_token in enc.encoder:
                    res_val_list.append(str(enc.encoder[bpe_token]))
                else:
                    res_val_list.append(str(enc.encoder["empty"]))


            # res_val_list = [str(enc.encoder[bpe_token]) for bpe_token in line_val_list]
            fvalo.write(" ".join(res_val_list) + "\n")



            line_sum = line_sum.strip()
            line_sum = line_sum.replace("-lrb-", "(")
            line_sum = line_sum.replace("-rrb-", ")")
            res_sum_list, _ = enc.encode(line_sum)
            fsumo.write(" ".join([str(tmp) for tmp in res_sum_list]) + "\n")


        fsumo.close()
        fvalo.close()


def get_train_vocab_bpe(domain):
    '''
    get train vocab of gpt data. return the mask
    '''

    summary_in = root_path + domain + '/original_data/train.summary'
    out_vocab = root_path + domain + '/processed_data/vocab_local.txt'

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



def preprocess(domain):
    """
    We use a triple <f, p+, p-> to represent the field information of a token in the specific field. 
    p+&p- are the position of the token in that field counted from the begining and the end of the field.
    For example, for a field (birthname, Jurgis Mikelatitis) in an infoboxes, we represent the field as
    (Jurgis, <birthname, 1, 2>) & (Mikelatitis, <birthname, 2, 1>)
    """
    print("extracting token, field type and position info from original data ...")
    time_start = time.time()
    split_infobox(domain)
    reverse_pos(domain)
    duration = time.time() - time_start
    print("extract finished in %.3f seconds" % float(duration))

    print("spliting test and valid summaries for ROUGE evaluation ...")
    time_start = time.time()
    split_summary_for_rouge(domain)
    duration = time.time() - time_start
    print("split finished in %.3f seconds" % float(duration))

    print("turning words and field types to ids ...")
    time_start = time.time()
    table2id(domain)
    duration = time.time() - time_start
    print("idlization finished in %.3f seconds" % float(duration))

    print("get vocab for train set")
    get_train_vocab_bpe(domain)



def make_dirs(domain):
    os.mkdir(root_path + domain + "/results/")
    os.mkdir(root_path + domain + "/results/res/")
    os.mkdir(root_path + domain + "/results/evaluation/")
    os.mkdir(root_path + domain + "/processed_data/")
    os.mkdir(root_path + domain + "/processed_data/train/")
    os.mkdir(root_path + domain + "/processed_data/test/")
    os.mkdir(root_path + domain + "/processed_data/valid/")
    os.mkdir(root_path + domain + "/processed_data/test/test_split_for_rouge/")
    os.mkdir(root_path + domain + "/processed_data/valid/valid_split_for_rouge/")

if __name__ == '__main__':
    domain = sys.argv[1]
    make_dirs(domain)
    preprocess(domain)
    check_generated_box(domain)
    print("check done")













    
