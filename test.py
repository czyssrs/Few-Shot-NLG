#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu

import sys
import os
import tensorflow as tf
import time
from SeqUnit import *
from DataLoader import DataLoader, DataLoader_test
import numpy as np
###
from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import * 

#############
'''
Run test for each domain data
'''
#############

root_path = "../emb_pointer_copyloss_am/"

### data
# tf.app.flags.DEFINE_string("dir", root_path + "processed_data", 'data set directory')
# tf.app.flags.DEFINE_string("seed_dir", root_path + "domain_descriptions_processed_data/", "domain seed dir")
# tf.app.flags.DEFINE_string("source_seed", "humans_single", "source seed dir")
# tf.app.flags.DEFINE_string("books_seed", "books_single", "target seed dir")

tf.app.flags.DEFINE_string("load",'9','load directory') # BBBBBESTOFAll

tf.app.flags.DEFINE_string("songs_seed", "songs", "target seed dir")
tf.app.flags.DEFINE_string("films_seed", "films", "target seed dir")

tf.app.flags.DEFINE_boolean("use_coverage", True,'use coverage or not')
tf.app.flags.DEFINE_float("coverage_penalty", 1.0,'coverage loss penalty')

tf.app.flags.DEFINE_boolean("use_copy_gate", True,'use copy gate or not')
tf.app.flags.DEFINE_float("copy_gate_penalty", 0.1, 'copy gate loss penalty')

tf.app.flags.DEFINE_integer("extend_vocab_size", 200,'extended vocabulary size for oov')

tf.app.flags.DEFINE_float("rnet_penalty", 100.0, 'copy gate loss penalty')
tf.app.flags.DEFINE_integer("seed_round", 200, "train seed batch every n round of source")
tf.app.flags.DEFINE_integer("seed_epoch", 1, "train seed data for n epoch each time")



tf.app.flags.DEFINE_string("mode",'train','train or test')
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')

tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')
tf.app.flags.DEFINE_boolean("fgate_encoder", True,'add field gate in encoder lstm')

tf.app.flags.DEFINE_boolean("field", False,'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position", False,'concat position information to word embedding')
tf.app.flags.DEFINE_boolean("encoder_pos", True,'position information in field-gated encoder')
tf.app.flags.DEFINE_boolean("decoder_pos", True,'position information in dual attention decoder')

tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 300, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 300, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 500, "Number of training epoch.")
tf.app.flags.DEFINE_integer("source_vocab", 5420,'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 2759,'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 5420,'vocabulary size')
tf.app.flags.DEFINE_integer("report", 1000,'report valid results after some steps')
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

tf.app.flags.DEFINE_integer("report_loss", 100,'report loss results after some steps')

FLAGS = tf.app.flags.FLAGS
last_best = 0.0

domain_data_name = sys.argv[1]
model_dir = sys.argv[2]

### path for calculate ROUGE
# gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
# gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'

### data paths
root_path = "../emb_pointer_copyloss_am/"
domain_data_path = root_path + "other_domain_data/" + domain_data_name
gold_path_test = domain_data_path + "/original_data/test.summary"
processed_data_path = domain_data_path + "/processed_data"

field_vocab_file = root_path + "human_books_songs_films_field_vocab.txt"
vocab_file = root_path + "human_books_songs_films_word_vocab_2000.txt"

word2vec_file = "/scratch/home/zhiyu/wiki2bio/other_data/glove.6B.300d.txt"

### vocab for test. in preprocess.py
v = Vocab(vocab_file, field_vocab_file)

### model paths
# test phase
model_path = root_path + "results/res/" + model_dir + "/loads/" + FLAGS.load + "/"
save_dir = domain_data_path + "/results/res/" + model_dir + "/"
pred_dir = domain_data_path + "/results/evaluation/" + model_dir + "/"
save_file_dir = save_dir + "files/"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

save_dir += (FLAGS.load + "/")
pred_dir += (FLAGS.load + "/")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)

if not os.path.exists(save_file_dir):
    os.mkdir(save_file_dir)

pred_path = pred_dir + "pred_summary_"
pred_beam_path = pred_dir + "beam_summary_"

log_file = save_dir + "log.txt"

                    


def test(sess, dataloader, model):
    return evaluate(sess, dataloader, model, save_dir, 'test')


def evaluate(sess, dataloader, model, ksave_dir, mode='test'):

    texts_path = processed_data_path + "/test.box.val"
    gold_path = gold_path_test
    evalset = dataloader.test_set
    oov_list = dataloader.test_oov_list
    
    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []
    
    k = 0
    for x in dataloader.batch_iter(evalset, oov_list, FLAGS.batch_size, False, 0):
        predictions, atts = model.generate(x, sess)
        this_oov_list = x['oov_map']
        atts = np.squeeze(atts)
        idx = 0
        for summary, oov_dict in zip(np.array(predictions), this_oov_list):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid >= FLAGS.target_vocab:
                        unk_sum.append("<_UNK_TOKEN_COPY>")
                        if tid in oov_dict:
                            real_sum.append(oov_dict[tid])
                        else:
                            real_sum.append("<_UNK_TOKEN_WRONG>")
                    elif tid == 3:
                        unk_sum.append("<_UNK_TOKEN_OOV>")
                        sub = texts[k][np.argmax(atts[tk,: len(texts[k]),idx])]
                        real_sum.append(sub)
                    else:
                        real_sum.append(v.id2word(tid))
                        unk_sum.append(v.id2word(tid))


                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                # pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1

    # write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")
    write_word(pred_list, ksave_dir, mode + "_summary_copy.clean.txt")


    ### new bleu
    bleu_unk = bleu_score(gold_path, ksave_dir + mode + "_summary_unk.txt")
    nocopy_result = "without copy BLEU: %.4f\n" % bleu_unk
    bleu_copy = bleu_score(gold_path, ksave_dir + mode + "_summary_copy.clean.txt")
    copy_result = "with copy BLEU: %.4f\n" % bleu_copy

    result = copy_result + nocopy_result 

    return result



def write_log(s):
    print s
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        copy_file(save_file_dir)

        init_word_emb = create_init_embedding(vocab_file, FLAGS.extend_vocab_size, word2vec_file, 300)
        assert len(init_word_emb) == (FLAGS.source_vocab + FLAGS.extend_vocab_size)

        # init_word_emb = None

        dataloader = DataLoader_test(processed_data_path, FLAGS.limits)
        field_id2word = dataloader.fieldid2word

        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                        field_size=FLAGS.field_size, pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                        source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
                        target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                        field_concat=FLAGS.field, position_concat=FLAGS.position,
                        fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention, decoder_add_pos=FLAGS.decoder_pos,
                        encoder_add_pos=FLAGS.encoder_pos, learning_rate=FLAGS.learning_rate,
                        use_coverage = FLAGS.use_coverage, coverage_penalty=FLAGS.coverage_penalty,
                        init_word_embedding = init_word_emb, extend_vocab_size=FLAGS.extend_vocab_size,
                        fieldid2word = field_id2word, use_glove=True, copy_gate_penalty=FLAGS.copy_gate_penalty,
                        use_copy_gate=FLAGS.use_copy_gate, rnet_penalty=FLAGS.rnet_penalty)


        sess.run(tf.global_variables_initializer())
        model.load(model_path)
        print "model loaded."
        write_log(test(sess, dataloader, model))


if __name__=='__main__':

    main()














