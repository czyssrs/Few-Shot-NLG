#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu

import sys
import os
import tensorflow as tf
import time
from SeqUnit import *
from DataLoader import DataLoader
import numpy as np
###
# from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import * 


tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 300, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 300, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")
tf.app.flags.DEFINE_integer("source_vocab", 5420,'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 2759,'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 5420,'vocabulary size')
tf.app.flags.DEFINE_integer("report", 5000,'report valid results after some steps')
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

tf.app.flags.DEFINE_boolean("use_coverage", False,'use coverage or not')
tf.app.flags.DEFINE_float("coverage_penalty", 0.1,'coverage loss penalty')

tf.app.flags.DEFINE_integer("extend_vocab_size", 30,'extended vocabulary size for oov')

tf.app.flags.DEFINE_string("mode",'train','train or test')
tf.app.flags.DEFINE_string("load",'0','load directory') # BBBBBESTOFAll
tf.app.flags.DEFINE_string("dir",'/scratch/home/zhiyu/wiki2bio/emb_baseline_pointer/processed_data','data set directory')
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')


tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')
tf.app.flags.DEFINE_boolean("fgate_encoder", True,'add field gate in encoder lstm')

tf.app.flags.DEFINE_boolean("field", False,'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position", False,'concat position information to word embedding')
tf.app.flags.DEFINE_boolean("encoder_pos", True,'position information in field-gated encoder')
tf.app.flags.DEFINE_boolean("decoder_pos", True,'position information in dual attention decoder')

tf.app.flags.DEFINE_integer("report_loss", 100,'report loss results after some steps')

FLAGS = tf.app.flags.FLAGS
last_best = 0.0

model_dir = sys.argv[1]

### path for calculate ROUGE
# gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
# gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'

###
# root_path = "/scratch/home/zhiyu/wiki2bio/"
# root_path = "../"
root_path = "../emb_baseline_pointer/"
gold_path_valid = root_path + 'original_data/valid.summary'
gold_path_test = root_path + 'original_data/test.summary'

field_vocab_file = root_path + "human_books_songs_films_field_vocab.txt"
vocab_file = root_path + "human_books_songs_films_word_vocab_2000.txt"

word2vec_file = "/scratch/home/zhiyu/wiki2bio/other_data/glove.42B.300d.zip"

# test phase
#### need to change!!!
if FLAGS.load != "0":
    save_dir = root_path + 'results/res/' + model_dir + '/loads/' + FLAGS.load + '/'
    save_file_dir = root_path + 'results/res/' + model_dir + '/' + 'files/'
    pred_dir = root_path + 'results/evaluation/' + model_dir + '/' + FLAGS.load + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'
# train phase
else:
    # prefix = str(int(time.time() * 1000))
    os.mkdir(root_path + 'results/res/' + model_dir)
    os.mkdir(root_path + 'results/evaluation/' + model_dir)
    save_dir = root_path + 'results/res/' + model_dir + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = root_path + 'results/evaluation/' + model_dir + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'

log_file = save_dir + 'log.txt'


def train(sess, dataloader, model):
    write_log("#######################################################")
    for flag in FLAGS.__flags:
        write_log(flag + " = " + str(FLAGS.__flags[flag]))
    write_log("#######################################################")
    trainset = dataloader.train_set
    oov_list = dataloader.train_oov_list
    k = 0
    record_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0
    record_cov_loss = 0.0
    for _ in range(FLAGS.epoch):
        for x in dataloader.batch_iter(trainset, oov_list, FLAGS.batch_size, True):
            this_loss, this_covloss = model(x, sess)
            loss += this_loss
            record_loss += this_loss
            record_cov_loss += this_covloss
            k += 1
            record_k += 1
            progress_bar(k%FLAGS.report, FLAGS.report)

            # ksave_dir = save_model(model, save_dir, k // FLAGS.report)
            # write_log(evaluate(sess, dataloader, model, ksave_dir, 'test'))
            ### czy
            if (record_k % FLAGS.report_loss == 0):
                write_log("%d : loss = %.3f, covloss = %.3f " % (k, record_loss / record_k, record_cov_loss / record_k))
                record_k = 0
                record_loss = 0.0
                record_cov_loss = 0.0

            if (k % FLAGS.report == 0):
                print "Round: ", k / FLAGS.report
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                if k // FLAGS.report >= 1: 
                    ksave_dir = save_model(model, save_dir, k // FLAGS.report)
                    write_log(evaluate(sess, dataloader, model, ksave_dir, 'valid'))
                    write_log(evaluate(sess, dataloader, model, ksave_dir, 'test'))
                    


def test(sess, dataloader, model):
    evaluate(sess, dataloader, model, save_dir, 'test')

def save_model(model, save_dir, cnt):
    new_dir = save_dir + 'loads' + '/' 
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    nnew_dir = new_dir + str(cnt) + '/'
    if not os.path.exists(nnew_dir):
        os.mkdir(nnew_dir)
    model.save(nnew_dir)
    return nnew_dir


def evaluate_old(sess, dataloader, model, ksave_dir, mode='valid'):
    '''
    original eva fn. with post processing to replace unk
    '''

    if mode == 'valid':
        # texts_path = "original_data/valid.summary"
        texts_path = root_path + "processed_data/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        # texts_path = "original_data/test.summary"
        texts_path = root_path + "processed_data/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set
    
    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []
    
    k = 0
    for x in dataloader.batch_iter(evalset, FLAGS.batch_size, False):
        predictions, atts = model.generate(x, sess)
        atts = np.squeeze(atts)
        idx = 0
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        sub = texts[k][np.argmax(atts[tk,: len(texts[k]),idx])]
                        real_sum.append(sub)
                        mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(v.id2word(tid))
                        mask_sum.append(v.id2word(tid))
                    unk_sum.append(v.id2word(tid))
                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1

    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")
    write_word(pred_list, ksave_dir, mode + "_summary_copy.clean.txt")


    ### new bleu
    # print ksave_dir + mode + "_summary_unk.txt"
    bleu_unk = bleu_score(gold_path, ksave_dir + mode + "_summary_unk.txt")
    nocopy_result = "without copy BLEU: %.4f\n" % bleu_unk
    bleu_copy = bleu_score(gold_path, ksave_dir + mode + "_summary_copy.clean.txt")
    copy_result = "with copy BLEU: %.4f\n" % bleu_copy


    # ### old bleu. too slow
    # for tk in range(k):
    #     with open(gold_path + str(tk), 'r') as g:
    #         gold_list.append([g.read().strip().split()])

    # gold_set = [[gold_path + str(i)] for i in range(k)]
    # pred_set = [pred_path + str(i) for i in range(k)]

    # # recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    # # bleu = corpus_bleu(gold_list, pred_list)
    # # copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    # # (str(F_measure), str(recall), str(precision), str(bleu))
    # # print copy_result

    # bleu = corpus_bleu(gold_list, pred_list)
    # copy_result = "with copy BLEU: %s\n" % (str(bleu))
    # #print copy_result

    # for tk in range(k):
    #     with open(pred_path + str(tk), 'w') as sw:
    #         sw.write(" ".join(pred_unk[tk]) + '\n')

    # # recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    # # bleu = corpus_bleu(gold_list, pred_unk)
    # # nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    # # (str(F_measure), str(recall), str(precision), str(bleu))
    # # print nocopy_result

    # bleu = corpus_bleu(gold_list, pred_unk)
    # nocopy_result = "without copy BLEU: %s\n" % (str(bleu))
    #print nocopy_result

    result = copy_result + nocopy_result 
    print result
    # if mode == 'valid':
    #     print result

    return result


def evaluate(sess, dataloader, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        # texts_path = "original_data/valid.summary"
        texts_path = root_path + "processed_data/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
        oov_list = dataloader.dev_oov_list
    else:
        # texts_path = "original_data/test.summary"
        texts_path = root_path + "processed_data/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set
        oov_list = dataloader.test_oov_list
    
    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []
    
    k = 0
    for x in dataloader.batch_iter(evalset, oov_list, FLAGS.batch_size, False):
        predictions, atts = model.generate(x, sess)
        this_oov_list = x['oov_map']
        atts = np.squeeze(atts)
        idx = 0
        for summary, oov_dict in zip(np.array(predictions), this_oov_list):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum = []
                unk_sum = []
                for tk, tid in enumerate(summary):
                    if tid >= FLAGS.target_vocab:
                        unk_sum.append("<_UNK_TOKEN>")
                        if tid in oov_dict:
                            real_sum.append(oov_dict[tid])
                        else:
                            real_sum.append("<_UNK_TOKEN>")
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
    # print ksave_dir + mode + "_summary_unk.txt"
    bleu_unk = bleu_score(gold_path, ksave_dir + mode + "_summary_unk.txt")
    nocopy_result = "without copy BLEU: %.4f\n" % bleu_unk
    bleu_copy = bleu_score(gold_path, ksave_dir + mode + "_summary_copy.clean.txt")
    copy_result = "with copy BLEU: %.4f\n" % bleu_copy


    # ### old bleu. too slow
    # for tk in range(k):
    #     with open(gold_path + str(tk), 'r') as g:
    #         gold_list.append([g.read().strip().split()])

    # gold_set = [[gold_path + str(i)] for i in range(k)]
    # pred_set = [pred_path + str(i) for i in range(k)]

    # # recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    # # bleu = corpus_bleu(gold_list, pred_list)
    # # copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    # # (str(F_measure), str(recall), str(precision), str(bleu))
    # # print copy_result

    # bleu = corpus_bleu(gold_list, pred_list)
    # copy_result = "with copy BLEU: %s\n" % (str(bleu))
    # #print copy_result

    # for tk in range(k):
    #     with open(pred_path + str(tk), 'w') as sw:
    #         sw.write(" ".join(pred_unk[tk]) + '\n')

    # # recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    # # bleu = corpus_bleu(gold_list, pred_unk)
    # # nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    # # (str(F_measure), str(recall), str(precision), str(bleu))
    # # print nocopy_result

    # bleu = corpus_bleu(gold_list, pred_unk)
    # nocopy_result = "without copy BLEU: %s\n" % (str(bleu))
    #print nocopy_result

    result = copy_result + nocopy_result 
    print result
    # if mode == 'valid':
    #     print result

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

        # init_word_emb = create_init_embedding(vocab_file, FLAGS.extend_vocab_size, word2vec_file, 300)
        # assert len(init_word_emb) == (FLAGS.source_vocab + FLAGS.extend_vocab_size)

        init_word_emb = None

        dataloader = DataLoader(FLAGS.dir, FLAGS.limits)
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
                        fieldid2word = field_id2word, use_glove=False)

        sess.run(tf.global_variables_initializer())
        # copy_file(save_file_dir)
        if FLAGS.load != '0':
            model.load(save_dir)
        if FLAGS.mode == 'train':
            train(sess, dataloader, model)
        else:
            test(sess, dataloader, model)


if __name__=='__main__':
    # with tf.device('/gpu:' + FLAGS.gpu):
    main()
