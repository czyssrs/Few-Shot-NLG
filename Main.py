#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import tensorflow as tf
import time
from SeqUnit import *
from DataLoader import DataLoader
import numpy as np
import model as model_gpt
from tqdm import tqdm
###
# from PythonROUGE import PythonROUGE
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from preprocess import *
import encoder
import json
from util import * 


tf.app.flags.DEFINE_string("gpt_model_name",'117M','model name of gpt2')
tf.app.flags.DEFINE_string("domain",'humans','domain name')

tf.app.flags.DEFINE_boolean("use_coverage", True,'use coverage or not')
tf.app.flags.DEFINE_float("coverage_penalty", 2.0,'coverage loss penalty')

tf.app.flags.DEFINE_boolean("use_copy_gate", True,'use copy gate or not')
tf.app.flags.DEFINE_float("copy_gate_penalty", 0.01, 'copy gate loss penalty')

tf.app.flags.DEFINE_string("mode",'train','train or test')
tf.app.flags.DEFINE_string("load",'0','load directory') # BBBBBESTOFAll
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')

tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')
tf.app.flags.DEFINE_boolean("fgate_encoder", True,'add field gate in encoder lstm')

tf.app.flags.DEFINE_boolean("field", False,'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position", False,'concat position information to word embedding')
tf.app.flags.DEFINE_boolean("encoder_pos", True,'position information in field-gated encoder')
tf.app.flags.DEFINE_boolean("decoder_pos", True,'position information in dual attention decoder')

tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 768, "Size of embedding.") # embedding for gpt
tf.app.flags.DEFINE_integer("field_size", 768, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 500, "Number of training epoch.")
tf.app.flags.DEFINE_integer("source_vocab", 50257,'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 2756,'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 50257,'vocabulary size')
tf.app.flags.DEFINE_integer("report", 500,'report valid results after some steps')
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

tf.app.flags.DEFINE_integer("report_loss", 10,'report loss results after some steps')

FLAGS = tf.app.flags.FLAGS
last_best = 0.0

model_dir = sys.argv[1]

### path for calculate ROUGE
# gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
# gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'


###
root_path = "../few_shot_gpt-2_data/"
gold_path_valid = root_path + FLAGS.domain + '/original_data/valid.summary'
gold_path_test = root_path + FLAGS.domain + '/original_data/test.summary'

field_vocab_file = root_path + "human_books_songs_films_field_vocab.txt"
vocab_file = root_path + "human_books_songs_films_word_vocab_2000.txt"

# word2vec_file = "/scratch/home/zhiyu/wiki2bio/other_data/glove.6B.300d.txt"
processed_data_dir = root_path + FLAGS.domain + "/processed_data"

### bpe vocab
enc = encoder.get_encoder("117M")
# "<|endoftext|>": 50256
eos = 50256
empty = 28920


# test phase
#### need to change!!!
if FLAGS.mode == "test":
    save_dir = root_path + FLAGS.domain +'/results/res/' + model_dir + '/loads/' + FLAGS.load + '/'
    save_file_dir = root_path + FLAGS.domain +'/results/res/' + model_dir + '/' + 'files/'
    pred_dir = root_path + FLAGS.domain +'/results/evaluation/' + model_dir + '/' + FLAGS.load + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'
# train phase
else:
    os.mkdir(root_path + FLAGS.domain +'/results/res/' + model_dir)
    os.mkdir(root_path + FLAGS.domain +'/results/evaluation/' + model_dir)
    save_dir = root_path + FLAGS.domain +'/results/res/' + model_dir + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = root_path + FLAGS.domain +'/results/evaluation/' + model_dir + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'

log_file = save_dir + 'log.txt'


def train(sess, dataloader, model):
    write_log("#######################################################")
    # print (FLAGS.flag_values_dict())
    for attr in FLAGS.flag_values_dict():
        value = FLAGS.flag_values_dict()[attr]
        write_log(attr + " = " + str(value))
    write_log("#######################################################")

    trainset = dataloader.train_set

    k = 0
    record_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0
    record_copy_loss = 0.0
    record_cov_loss = 0.0

    for _ in range(FLAGS.epoch):
        for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True, domain=FLAGS.domain):
            this_loss, this_copy_gate_loss, this_cov_loss = model(x, sess)
            loss += this_loss
            record_loss += this_loss
            record_copy_loss += this_copy_gate_loss
            record_cov_loss += this_cov_loss
            k += 1
            record_k += 1
            progress_bar(k % FLAGS.report, FLAGS.report)

            # ksave_dir = save_model(model, sess, save_dir, k // FLAGS.report)
            # write_log(evaluate(sess, dataloader, model, ksave_dir, 'test'))

            if (record_k % FLAGS.report_loss == 0):
                write_log("%d : loss = %.3f, copyloss = %.3f, covloss = %.3f" % \
                    (k, record_loss / record_k, record_copy_loss / record_k, record_cov_loss / record_k))
                record_k = 0
                record_loss = 0.0
                record_copy_loss = 0.0
                record_cov_loss = 0.0


            if (k % FLAGS.report == 0):
                print("Round: ", k / FLAGS.report)
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                if k // FLAGS.report >= 1: 
                    ksave_dir = save_model(model, sess, save_dir, k // FLAGS.report)
                    # write_log(evaluate(sess, dataloader, model, ksave_dir, 'valid'))
                    write_log(evaluate(sess, dataloader, model, ksave_dir, 'test'))
                    


def test(sess, dataloader, model):
    evaluate(sess, dataloader, model, save_dir, 'test')

def save_model(model, sess, save_dir, cnt):
    new_dir = save_dir + 'loads' + '/' 
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    nnew_dir = new_dir + str(cnt) + '/'
    if not os.path.exists(nnew_dir):
        os.mkdir(nnew_dir)
    model.save(nnew_dir, sess)
    return nnew_dir


def evaluate(sess, dataloader, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        texts_path = processed_data_dir + "/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        texts_path = processed_data_dir + "/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set

    pred_list = []
    pred_unk = []

    out_bpe = open(ksave_dir + mode + "_summary_bpe.txt", "w")
    out_real = open(ksave_dir + mode + "_summary.clean.txt", "w")
    
    k = 0
    for x in tqdm(dataloader.batch_iter(evalset, FLAGS.batch_size, False, domain=FLAGS.domain)):
        predictions, atts = model.generate(x, sess)
        # atts = np.squeeze(atts)
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                # print len(summary)
                if eos in summary:
                    summary = summary[:summary.index(eos)] if summary[0] != eos else [eos]
                real_sum = enc.decode(summary)
                bpe_sum = " ".join([enc.decoder[tmp] for tmp in summary])
                
                sw.write(real_sum + '\n')
                pred_list.append(real_sum)
                pred_unk.append(bpe_sum)

                out_real.write(real_sum + '\n')
                out_bpe.write(bpe_sum + '\n')

                # print (real_sum)
                k += 1

    # write_word(pred_unk, ksave_dir, mode + "_summary_bpe.txt")
    # write_word(pred_list, ksave_dir, mode + "_summary.clean.txt")
    out_bpe.close()
    out_real.close()

    ### new bleu
    bleu_copy = bleu_score(gold_path, ksave_dir + mode + "_summary.clean.txt")
    copy_result = "with copy BLEU: %.4f\n" % bleu_copy

    result = copy_result 

    return result

def write_log(s):
    print(s)

    with open(log_file, 'a') as f:
        f.write(s+'\n')


def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        copy_file(save_file_dir)

        hparams = model_gpt.default_hparams()
        with open(os.path.join('../models', FLAGS.gpt_model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        dataloader = DataLoader(processed_data_dir, FLAGS.limits, eos, empty)
        field_id2word = dataloader.fieldid2word

        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                        field_size=FLAGS.field_size, pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                        source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
                        target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                        field_concat=FLAGS.field, position_concat=FLAGS.position,
                        fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention, decoder_add_pos=FLAGS.decoder_pos,
                        encoder_add_pos=FLAGS.encoder_pos, learning_rate=FLAGS.learning_rate,
                        use_coverage = FLAGS.use_coverage, coverage_penalty=FLAGS.coverage_penalty,
                        fieldid2word = field_id2word, copy_gate_penalty=FLAGS.copy_gate_penalty,
                        use_copy_gate=FLAGS.use_copy_gate, gpt_hparams=hparams)


        if FLAGS.mode == 'train':
            ### load pretrained gpt
            gpt_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')

            # gpt_var_opt = []
            # for each_var in gpt_var:
            #     if "Adam" in each_var.name:
            #         gpt_var_opt.append(each_var)

            gpt_var_load = []
            for each_var in gpt_var:
                if "Adam" not in each_var.name:
                    gpt_var_load.append(each_var)


            # print ([tmp.name for tmp in gpt_var_load])
            saver = tf.train.Saver(var_list=gpt_var_load)
            ckpt = tf.train.latest_checkpoint(os.path.join('../models', FLAGS.gpt_model_name))
            saver.restore(sess, ckpt)

            # ### init other vars
            seq2seq_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq2seq')
            # print ([tmp.name for tmp in seq2seq_var])
            # seq2seq_var += gpt_var_opt

            sess.run(tf.variables_initializer(var_list=seq2seq_var))

            train(sess, dataloader, model)

        else:
            model.load(model_path, sess)
            write_log(test(sess, dataloader, model))


if __name__=='__main__':
    # with tf.device('/gpu:' + FLAGS.gpu):
    main()
