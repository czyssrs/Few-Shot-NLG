import os
import sys
import operator
import json
import random
import encoder
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


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


def get_train_vocab(box_in, summary_in, out_vocab):
    """
    get vocab for few shot baselines
    Args:
        box_in: str, path to input box file
        summary_in: str, path to input summary file
        out_vocab: str, path to output vocabulary file

    Returns:

    """
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

    print(len(sorted_x))
    print(ind)



def get_train_vocab_bpe_mask(summary_in, out_vocab):
    """
    get train vocab of gpt data. return the mask
    Args:
        summary_in:
        out_vocab:

    Returns:

    """
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

    print(len(vocab))

    res_mask = []
    for ind in range(0, 50257):
        if ind in vocab:
            res_mask.append(str(1))
        else:
            res_mask.append(str(0))

    with open(out_vocab, "w") as f:
        f.write(" ".join(res_mask))


def get_train_vocab_bpe(summary_in, box_in, json_ori_in, json_out, vocab_ind_out):
    """
    get train vocab of gpt data. return the mask
    Args:
        summary_in: str, file path for summary
        box_in: str, file path for box
        json_ori_in:
        json_out:
        vocab_ind_out:

    Returns:

    """

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

    print(len(vocab))


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

    print(len(out_vocab))
    print(out_vocab["empty"])
    print(out_vocab["<|endoftext|>"])

    with open(json_out, "w") as f:
        f.write(json.dumps(out_vocab))

    with open(vocab_ind_out, "w") as f:
        for ind in res_vocab:
            f.write(str(ind) + "\n")

def check_encoding():
    """
    Test GPT encoder
    Returns:
        None
    """

    test1 = [7839, 1058, 4171, 2933, 837, 1772, 1058, 474, 11025, 308, 295, 78, 837, 2656, 3670,
             1058, 474 ,11025, 443, 7245, 84 ,837 ,33417, 1058, 479 ,15289,257 ,13 ,10212, 365,
             837 ,1499 , 1058 ,1216, 590 ,837, 3303, 1058, 48718, 837 ,9991 ,1058 ,38251, 67 ,1756,
             8701, 316, 837 ,9207, 3128, 1058,32471, 837, 3199 ,287 ,46932, 1058 ,22717 ,837, 5468,
             1058 ,34131, 837]

    enc = encoder.get_encoder("117M")
    dec_out = enc.decode(test1)
    print(dec_out)


if __name__=='__main__':
    """
    Test all functions
    """
    root_path = sys.argv[1]
    model_path = sys.argv[2]
    domain = sys.argv[3]

    box_in = os.path.join(root_path, domain, 'original_data', 'test.box')
    summary_in = os.path.join(root_path, domain, 'original_data', 'test.summary')

    box_out = os.path.join(root_path, domain, 'original_data', 'test_full.box')
    summary_out = os.path.join(root_path, domain, 'original_data', 'test_full.summary')

    convert_bpe(box_in, summary_in, box_out, summary_out)

    out_vocab = os.path.join(root_path, domain, 'original_data', 'test_vocab.txt')

    get_train_vocab(box_in, summary_in, out_vocab)

    json_ori_in = os.path.join(model_path, '117M_original', 'encoder.json')
    json_out = os.path.join(model_path, '117M', 'encoder.json')
    vocab_ind_out = os.path.join(model_path, '117M_original', 'vocab_ind.txt')
    get_train_vocab_bpe(summary_in, box_in, json_ori_in, json_out, vocab_ind_out)

    check_encoding()














