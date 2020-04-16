#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for running few shot training
Eg.
python ./Main.py --root_path ~/Data/NLP/few_shot_nlg/ --domain humans --gpt_model_name ../models/117M/ --output_path ~/Output/
"""
from SeqUnit import *
from DataLoader import DataLoader, Preprocessor
import model as model_gpt
from tqdm import tqdm
import encoder
import json
from util import get_current_git_version, bleu_score, write_log
from datetime import datetime
import time


# paths and mode
tf.app.flags.DEFINE_string("root_path", "../data_release/", "full path of data folder")
tf.app.flags.DEFINE_string("domain",'humans','domain name')
tf.app.flags.DEFINE_string("gpt_model_name",'../models/117M','full path of gpt2 model')
tf.app.flags.DEFINE_string("output_path", "../tmp/", "full path of saved output")
tf.app.flags.DEFINE_string("mode",'train','train or test')
tf.app.flags.DEFINE_string("saved_model_path",'temp','saved model path for use in test mode')

# architecture choices
tf.app.flags.DEFINE_boolean("use_coverage", False,'use coverage or not')
tf.app.flags.DEFINE_float("coverage_penalty", 0.02,'coverage loss penalty')
tf.app.flags.DEFINE_boolean("use_copy_gate", True,'use copy gate or not')
tf.app.flags.DEFINE_float("copy_gate_penalty", 0.7, 'copy gate loss penalty')
tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')
tf.app.flags.DEFINE_boolean("fgate_encoder", True,'add field gate in encoder lstm')

# data options
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')
tf.app.flags.DEFINE_boolean("field", False,'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position", False,'concat position information to word embedding')
tf.app.flags.DEFINE_boolean("encoder_pos", True,'position information in field-gated encoder')
tf.app.flags.DEFINE_boolean("decoder_pos", True,'position information in dual attention decoder')
tf.app.flags.DEFINE_integer("source_vocab", 50257,'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 2756,'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 50257,'vocabulary size')

# model hyperparams
tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 768, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 768, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")

# training
tf.app.flags.DEFINE_integer("batch_size", 2, "Batch size of train set.")
tf.app.flags.DEFINE_integer("batch_update", 22, "apply gradients after steps")
tf.app.flags.DEFINE_integer("epoch", 5000, "Number of training epoch.")
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

# logging
tf.app.flags.DEFINE_integer("report", 30,'report valid results after some steps')
tf.app.flags.DEFINE_integer("report_loss", 10,'report loss results after some steps')

FLAGS = tf.app.flags.FLAGS

# create output paths
if FLAGS.mode == "train":
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(FLAGS.output_path, model_dir_name)
    results_path = os.path.join(model_dir, FLAGS.domain, "results")
    saved_model_path = os.path.join(model_dir, FLAGS.domain, "saved_model")
else:
    saved_model_path = tf.app.flags.FLAGS.saved_model_path
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(FLAGS.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, FLAGS.domain, "results")

os.makedirs(results_path, exist_ok=False)
os.makedirs(saved_model_path, exist_ok=False)
log_file = os.path.join(results_path, 'log.txt')


# create data paths
root_path = FLAGS.root_path
gold_path_valid = os.path.join(root_path, FLAGS.domain, 'original_data', 'valid.summary')
gold_path_test = os.path.join(root_path, FLAGS.domain, 'original_data', 'test.summary')
field_vocab_file = os.path.join(root_path, "human_books_songs_films_field_vocab.txt")
processed_data_dir = os.path.join(root_path, FLAGS.domain, "processed_data")

# bpe vocab
last_best = 0.0
enc = encoder.get_encoder("117M")
eos = 50256 #TODO move to settings
empty = 2 #TODO move to settings


def train(sess, preprocessed_data, model):
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in FLAGS.flag_values_dict():
        value = FLAGS.flag_values_dict()[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    train_iterator = DataLoader(preprocessed_data.train_set, FLAGS.domain,
                                batch_size=FLAGS.batch_size, shuffle=True, eos=eos, empty=empty)

    k = 0
    record_k = 0
    record_loss_k = 0 
    loss, start_time = 0.0, time.time()
    record_loss = 0.0
    record_copy_loss = 0.0
    record_cov_loss = 0.0

    for _ in range(FLAGS.epoch):
        train_iterator.reset()
        for x in train_iterator:
            model(x, sess, 0)
            k += 1

            #TODO also add to tensorboard
            if k % FLAGS.batch_update == 0:
                this_loss, this_copy_gate_loss, this_cov_loss = model(x, sess, 1)
                record_loss += this_loss
                record_copy_loss += this_copy_gate_loss
                record_cov_loss += this_cov_loss
                record_k += 1
                record_loss_k += 1

                if record_loss_k > 1 and record_loss_k % FLAGS.report_loss == 0:
                    write_log(log_file, "%d : loss = %.3f, copyloss = %.3f, covloss = %.3f" % \
                        (record_k, record_loss / record_loss_k, record_copy_loss / record_loss_k,
                         record_cov_loss / record_loss_k))
                    record_loss = 0.0
                    record_copy_loss = 0.0
                    record_cov_loss = 0.0
                    record_loss_k = 0

                if record_k > 1 and record_k % FLAGS.report == 0:
                    print("Round: ", record_k / FLAGS.report)
                    cost_time = time.time() - start_time
                    write_log(log_file, "%d : time = %.3f " % (record_k // FLAGS.report, cost_time))
                    start_time = time.time()
                    if record_k // FLAGS.report >= 1:
                        # save model
                        saved_model_path_cnt = os.path.join(saved_model_path, 'loads', str(record_k // FLAGS.report))
                        os.makedirs(saved_model_path_cnt, exist_ok=True)
                        model.save(saved_model_path_cnt, sess)

                        results_path_cnt = os.path.join(results_path, 'loads', str(record_k // FLAGS.report))
                        os.makedirs(results_path_cnt, exist_ok=True)
                        validation_result = evaluate(sess, preprocessed_data, model, results_path_cnt, 'valid')
                        write_log(log_file, validation_result)


def evaluate(sess, preprocessed_data, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        gold_path = gold_path_valid
        data_iterator = DataLoader(preprocessed_data.dev_set,
                                    FLAGS.domain, batch_size=FLAGS.batch_size, shuffle=False,
                                    eos=eos, empty=empty)
    else:
        gold_path = gold_path_test
        data_iterator = DataLoader(preprocessed_data.test_set,
                                   FLAGS.domain, batch_size=FLAGS.batch_size, shuffle=False, eos=eos,
                                   empty=empty)

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    out_bpe = open(os.path.join(ksave_dir_mode, mode + "_summary_bpe.txt"), "w")
    out_real = open(os.path.join(ksave_dir_mode,  mode + "_summary.clean.txt"), "w")
    pred_path = os.path.join(ksave_dir_mode,  mode + "_pred_summary_")

    k = 0
    for x in tqdm(data_iterator):
        predictions, atts = model.generate(x, sess)

        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)

                if eos in summary:
                    summary = summary[:summary.index(eos)] if summary[0] != eos else [eos]
                real_sum = enc.decode(summary)
                bpe_sum = " ".join([enc.decoder[tmp] for tmp in summary])

                real_sum = real_sum.replace("\n", " ")
                sw.write(real_sum + '\n')
                pred_list.append(real_sum)
                pred_unk.append(bpe_sum)

                out_real.write(real_sum + '\n')
                out_bpe.write(bpe_sum + '\n')

                k += 1

    out_bpe.close()
    out_real.close()

    # new bleu
    bleu_copy = bleu_score(gold_path, os.path.join(ksave_dir_mode,  mode + "_summary.clean.txt"))
    copy_result = "with copy BLEU: %.4f\n" % bleu_copy

    result = copy_result

    return result


def main():

    # keep track of the commit id
    git_commit_id = get_current_git_version()
    write_log(log_file, "GIT COMMIT ID: " + git_commit_id)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=tf.Graph()) as sess:
        hparams = model_gpt.default_hparams()
        with open(os.path.join(FLAGS.gpt_model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        preprocessed_data = Preprocessor(processed_data_dir, FLAGS.limits, eos, empty)
        field_id2word = preprocessed_data.fieldid2word

        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size,
                        emb_size=FLAGS.emb_size, field_size=FLAGS.field_size,
                        pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                        source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
                        target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                        field_concat=FLAGS.field, position_concat=FLAGS.position,
                        fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention,
                        decoder_add_pos=FLAGS.decoder_pos, encoder_add_pos=FLAGS.encoder_pos,
                        learning_rate=FLAGS.learning_rate, use_coverage = FLAGS.use_coverage,
                        coverage_penalty=FLAGS.coverage_penalty, fieldid2word = field_id2word,
                        copy_gate_penalty=FLAGS.copy_gate_penalty, use_copy_gate=FLAGS.use_copy_gate,
                        gpt_hparams=hparams, vocab_ind=None,
                        empty_token=empty, stop_token=eos)

        if FLAGS.mode == 'train':
            # collect all trainable variables, exclude embeddings
            gpt_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
            gpt_var_load = []
            for each_var in gpt_var:
                if "Adam" not in each_var.name:
                    gpt_var_load.append(each_var)
            gpt_var_load.remove(model.embedding)

            # load GPT checkpoint
            saver = tf.train.Saver(var_list=gpt_var_load)
            ckpt = tf.train.latest_checkpoint(FLAGS.gpt_model_name)
            saver.restore(sess, ckpt)

            # init other vars
            seq2seq_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq2seq')
            seq2seq_var.append(model.embedding)
            sess.run(tf.variables_initializer(var_list=seq2seq_var))

            train(sess, preprocessed_data, model)

        else:
            model.load(saved_model_path, sess)
            test_result = evaluate(sess, preprocessed_data, model, results_path, 'test')
            write_log(log_file, test_result)


if __name__ == '__main__':
    main()
