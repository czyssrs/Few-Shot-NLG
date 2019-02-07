#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:43
# @Author  : Tianyu Liu

import tensorflow as tf
import time
import numpy as np


class DataLoader(object):
    def __init__(self, data_dir, limits):
        self.train_data_path = [data_dir + '/train/train.summary.id', data_dir + '/train/train.box.val.id',
                                data_dir + '/train/train.box.lab.id', data_dir + '/train/train.box.pos',
                                data_dir + '/train/train.box.rpos', data_dir + '/train/train_summary_field_id.txt',
                                data_dir + '/train/train_summary_pos.txt', data_dir + '/train/train_summary_rpos.txt']

        self.test_data_path = [data_dir + '/test/test.summary.id', data_dir + '/test/test.box.val.id',
                               data_dir + '/test/test.box.lab.id', data_dir + '/test/test.box.pos',
                               data_dir + '/test/test.box.rpos', data_dir + '/test/test_summary_field_id.txt',
                                data_dir + '/test/test_summary_pos.txt', data_dir + '/test/test_summary_rpos.txt']

        self.dev_data_path = [data_dir + '/valid/valid.summary.id', data_dir + '/valid/valid.box.val.id',
                              data_dir + '/valid/valid.box.lab.id', data_dir + '/valid/valid.box.pos',
                              data_dir + '/valid/valid.box.rpos', data_dir + '/valid/valid_summary_field_id.txt',
                                data_dir + '/valid/valid_summary_pos.txt', data_dir + '/valid/valid_summary_rpos.txt']

        self.train_local_oov_path = data_dir + "/train/train_local_oov.txt"
        self.dev_local_oov_path = data_dir + "/valid/valid_local_oov.txt"
        self.test_local_oov_path = data_dir + "/test/test_local_oov.txt"

        self.limits = limits
        self.man_text_len = 100
        self.man_summary_len = 150
        start_time = time.time()

        print('Reading datasets ...')
        self.train_set = self.load_data(self.train_data_path)
        self.test_set = self.load_data(self.test_data_path)
        # self.small_test_set = self.load_data(self.small_test_data_path)
        self.dev_set = self.load_data(self.dev_data_path)
        print ('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))

        ### load fieldid2word list len 3
        self.fieldid2word = []
        with open(data_dir + "/field2word.txt") as f:
            for line in f:
                fieldid = int(line.strip().split("\t")[0])
                word_list = line.strip().split("\t")[1].split(" ")
                wordid_list = [int(tmp) for tmp in word_list]
                assert len(wordid_list) == 3

                self.fieldid2word.append(wordid_list)

        self.fieldid2word = np.array(self.fieldid2word)

        self.dev_oov_list = self.load_local_oov(self.dev_local_oov_path)
        self.test_oov_list = self.load_local_oov(self.test_local_oov_path)
        self.train_oov_list = self.load_local_oov(self.train_local_oov_path)


    def load_local_oov(self, path):
        '''
        load oov map for each example
        '''
        oov_list = []
        with open(path) as f:
            for line in f:
                if line.strip() == "":
                    oov_list.append({})
                    continue
                line_list = line.strip().split("\t")
                this_oov_dict = {}
                for item in line_list:
                    oov_id = int(item.split(":")[0])
                    oov_word = item.split(":")[1]

                    this_oov_dict[oov_id] = oov_word

                oov_list.append(this_oov_dict)

        return oov_list


    def load_data(self, path):
        summary_path, text_path, field_path, pos_path, rpos_path, dec_field, dec_pos, dec_rpos = path
        summaries = open(summary_path, 'r').read().strip().split('\n')
        texts = open(text_path, 'r').read().strip().split('\n')
        fields = open(field_path, 'r').read().strip().split('\n')
        poses = open(pos_path, 'r').read().strip().split('\n')
        rposes = open(rpos_path, 'r').read().strip().split('\n')
        decoder_field = open(dec_field).read().strip().split('\n')
        decoder_pos = open(dec_pos).read().strip().split('\n')
        decoder_rpos = open(dec_rpos).read().strip().split('\n')

        if self.limits > 0:
            summaries = summaries[:self.limits]
            texts = texts[:self.limits]
            fields = fields[:self.limits]
            poses = poses[:self.limits]
            rposes = rposes[:self.limits]
            decoder_field = decoder_field[:self.limits]
            decoder_pos = decoder_pos[:self.limits]
            decoder_rpos = decoder_rpos[:self.limits]

        print summaries[0].strip().split(' ')
        summaries = [list(map(int, summary.strip().split(' '))) for summary in summaries]
        texts = [list(map(int, text.strip().split(' '))) for text in texts]
        fields = [list(map(int, field.strip().split(' '))) for field in fields]
        poses = [list(map(int, pos.strip().split(' '))) for pos in poses]
        rposes = [list(map(int, rpos.strip().split(' '))) for rpos in rposes]
        decoder_field = [list(map(int, field.strip().split(' '))) for field in decoder_field]
        decoder_pos = [list(map(int, pos.strip().split(' '))) for pos in decoder_pos]
        decoder_rpos = [list(map(int, rpos.strip().split(' '))) for rpos in decoder_rpos]
        return summaries, texts, fields, poses, rposes, decoder_field, decoder_pos, decoder_rpos

    def batch_iter(self, data, oov_list, batch_size, shuffle):
        summaries, texts, fields, poses, rposes, decoder_field, decoder_pos, decoder_rpos = data
        data_size = len(summaries)
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            summaries = np.array(summaries)[shuffle_indices]
            texts = np.array(texts)[shuffle_indices]
            fields = np.array(fields)[shuffle_indices]
            poses = np.array(poses)[shuffle_indices]
            rposes = np.array(rposes)[shuffle_indices]
            
            oov_list = np.array(oov_list)[shuffle_indices]

            decoder_field = np.array(decoder_field)[shuffle_indices]
            decoder_pos = np.array(decoder_pos)[shuffle_indices]
            decoder_rpos = np.array(decoder_rpos)[shuffle_indices]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            max_summary_len = max([len(sample) for sample in summaries[start_index:end_index]])
            max_text_len = max([len(sample) for sample in texts[start_index:end_index]])
            batch_data = {'enc_in':[], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[], 'enc_len':[],
                          'dec_in':[], 'dec_len':[], 'dec_out':[], 'oov_map':[], 'dec_field':[],
                          'dec_pos':[], 'dec_rpos':[]}

            for summary, text, field, pos, rpos, oov, dec_field, dec_pos, dec_rpos in zip(summaries[start_index:end_index], texts[start_index:end_index],
                                            fields[start_index:end_index], poses[start_index:end_index],
                                            rposes[start_index:end_index], oov_list[start_index:end_index],
                                            decoder_field[start_index:end_index], decoder_pos[start_index:end_index],
                                            decoder_rpos[start_index:end_index]):
                summary_len = len(summary)
                text_len = len(text)
                pos_len = len(pos)
                rpos_len = len(rpos)
                assert text_len == len(field)
                assert pos_len == len(field)
                assert rpos_len == pos_len

                assert len(dec_field) == len(summary)

                gold = summary + [2] + [0] * (max_summary_len - summary_len)
                summary = summary + [0] * (max_summary_len - summary_len)

                text = text + [0] * (max_text_len - text_len)
                field = field + [0] * (max_text_len - text_len)
                pos = pos + [0] * (max_text_len - text_len)
                rpos = rpos + [0] * (max_text_len - text_len)

                dec_field = dec_field + [0] * (max_summary_len - summary_len)
                dec_pos = dec_pos + [0] * (max_summary_len - summary_len)
                dec_rpos = dec_rpos + [0] * (max_summary_len - summary_len)

                # assert len(dec_field) == len(summary)

                # ###
                # emb_field = []
                # for each_item in field:
                #     emb_field.append(self.fieldid2word[each_item])
                
                if max_text_len > self.man_text_len:
                    text = text[:self.man_text_len]
                    field = field[:self.man_text_len]

                    # ###
                    # emb_field = emb_field[:self.man_text_len]

                    pos = pos[:self.man_text_len]
                    rpos = rpos[:self.man_text_len]
                    text_len = min(text_len, self.man_text_len)

                ### OOM
                if max_summary_len > self.man_summary_len:
                    summary = summary[:self.man_summary_len - 1]
                    dec_field = dec_field[:self.man_summary_len - 1]
                    dec_pos = dec_pos[:self.man_summary_len - 1]
                    dec_rpos = dec_rpos[:self.man_summary_len - 1]


                    if gold[self.man_summary_len - 2] == 0:
                        gold = gold[:self.man_summary_len - 1] + [0]
                    else:
                        gold = gold[:self.man_summary_len - 1] + [2]
                    if summary_len > self.man_summary_len - 1:
                        summary_len = self.man_summary_len - 1

                
                batch_data['enc_in'].append(text)
                batch_data['enc_len'].append(text_len)

                batch_data['enc_fd'].append(field)
                # batch_data['enc_fd'].append(emb_field)

                batch_data['enc_pos'].append(pos)
                batch_data['enc_rpos'].append(rpos)
                batch_data['dec_in'].append(summary)
                batch_data['dec_len'].append(summary_len)
                batch_data['dec_out'].append(gold)

                batch_data['oov_map'].append(oov)

                batch_data['dec_field'].append(dec_field)
                batch_data['dec_pos'].append(dec_pos)
                batch_data['dec_rpos'].append(dec_rpos)

                # print np.shape(batch_data['enc_fd'])
                # print np.shape(batch_data['enc_pos'])

                # print batch_data['enc_fd']

                # print batch_data['dec_rpos'][0]
                # print batch_data['enc_rpos'][0]
  
                # for item in batch_data['dec_field']:
                #     print len(item)
            yield batch_data





class DataLoader_test(object):
    def __init__(self, data_dir, limits, root_path):
        self.test_data_path = [data_dir + 'test.summary.id', data_dir + 'test.box.val.id',
                               data_dir + 'test.box.lab.id', data_dir + 'test.box.pos',
                               data_dir + 'test.box.rpos']
        self.limits = limits
        self.man_text_len = 100
        self.man_summary_len = 150
        start_time = time.time()

        print('Reading datasets ...')
        self.test_set = self.load_data(self.test_data_path)
        print ('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))

        ### load fieldid2word list len 3
        self.fieldid2word = []
        with open(data_dir + "/field2word.txt") as f:
            for line in f:
                fieldid = int(line.strip().split("\t")[0])
                word_list = line.strip().split("\t")[1].split(" ")
                wordid_list = [int(tmp) for tmp in word_list]
                assert len(wordid_list) == 3

                self.fieldid2word.append(wordid_list)

        self.fieldid2word = np.array(self.fieldid2word)

    def load_data(self, path):
        summary_path, text_path, field_path, pos_path, rpos_path = path
        summaries = open(summary_path, 'r').read().strip().split('\n')
        texts = open(text_path, 'r').read().strip().split('\n')
        fields = open(field_path, 'r').read().strip().split('\n')
        poses = open(pos_path, 'r').read().strip().split('\n')
        rposes = open(rpos_path, 'r').read().strip().split('\n')
        if self.limits > 0:
            summaries = summaries[:self.limits]
            texts = texts[:self.limits]
            fields = fields[:self.limits]
            poses = poses[:self.limits]
            rposes = rposes[:self.limits]
        print summaries[0].strip().split(' ')
        summaries = [list(map(int, summary.strip().split(' '))) for summary in summaries]
        texts = [list(map(int, text.strip().split(' '))) for text in texts]
        fields = [list(map(int, field.strip().split(' '))) for field in fields]
        poses = [list(map(int, pos.strip().split(' '))) for pos in poses]
        rposes = [list(map(int, rpos.strip().split(' '))) for rpos in rposes]
        return summaries, texts, fields, poses, rposes

    def batch_iter(self, data, batch_size, shuffle):
        summaries, texts, fields, poses, rposes = data
        data_size = len(summaries)
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            summaries = np.array(summaries)[shuffle_indices]
            texts = np.array(texts)[shuffle_indices]
            fields = np.array(fields)[shuffle_indices]
            poses = np.array(poses)[shuffle_indices]
            rposes = np.array(rposes)[shuffle_indices]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            max_summary_len = max([len(sample) for sample in summaries[start_index:end_index]])
            max_text_len = max([len(sample) for sample in texts[start_index:end_index]])
            batch_data = {'enc_in':[], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[], 'enc_len':[],
                          'dec_in':[], 'dec_len':[], 'dec_out':[]}

            for summary, text, field, pos, rpos in zip(summaries[start_index:end_index], texts[start_index:end_index],
                                            fields[start_index:end_index], poses[start_index:end_index],
                                            rposes[start_index:end_index]):
                summary_len = len(summary)
                text_len = len(text)
                pos_len = len(pos)
                rpos_len = len(rpos)
                assert text_len == len(field)
                assert pos_len == len(field)
                assert rpos_len == pos_len
                gold = summary + [2] + [0] * (max_summary_len - summary_len)
                summary = summary + [0] * (max_summary_len - summary_len)

                text = text + [0] * (max_text_len - text_len)
                field = field + [0] * (max_text_len - text_len)
                pos = pos + [0] * (max_text_len - text_len)
                rpos = rpos + [0] * (max_text_len - text_len)

                # ###
                # emb_field = []
                # for each_item in field:
                #     emb_field.append(self.fieldid2word[each_item])
                
                if max_text_len > self.man_text_len:
                    text = text[:self.man_text_len]
                    field = field[:self.man_text_len]

                    # ###
                    # emb_field = emb_field[:self.man_text_len]

                    pos = pos[:self.man_text_len]
                    rpos = rpos[:self.man_text_len]
                    text_len = min(text_len, self.man_text_len)

                ### OOM
                if max_summary_len > self.man_summary_len:
                    summary = summary[:self.man_summary_len - 1]
                    if gold[self.man_summary_len - 2] == 0:
                        gold = gold[:self.man_summary_len - 1] + [0]
                    else:
                        gold = gold[:self.man_summary_len - 1] + [2]
                    if summary_len > self.man_summary_len - 1:
                        summary_len = self.man_summary_len - 1
                
                batch_data['enc_in'].append(text)
                batch_data['enc_len'].append(text_len)

                batch_data['enc_fd'].append(field)
                # batch_data['enc_fd'].append(emb_field)

                batch_data['enc_pos'].append(pos)
                batch_data['enc_rpos'].append(rpos)
                batch_data['dec_in'].append(summary)
                batch_data['dec_len'].append(summary_len)
                batch_data['dec_out'].append(gold)

                # print np.shape(batch_data['enc_fd'])
                # print np.shape(batch_data['enc_pos'])
  
            yield batch_data






