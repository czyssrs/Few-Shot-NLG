#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:43
# @Author  : Tianyu Liu

import time
import numpy as np
import encoder
import os

enc = encoder.get_encoder("117M")


class DataLoader(object):
    def __init__(self, data_dir, limits, eos, empty):
        """
        Main dataloader
        Args:
            data_dir: str, path to data directory
            limits:
            eos: str, eos character
            empty:
        """
        self.data_dir = data_dir
        self.vocab_mask_path = data_dir + '/vocab_local.txt'

        self.limits = limits
        self.man_text_len = 150
        self.man_summary_len = 85
        self.eos = eos
        self.empty = empty
        start_time = time.time()

        print('Reading datasets ...')
        self.train_set = self.load_data('train')
        self.test_set = self.load_data('test')
        self.dev_set = self.load_data('valid')
        print('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))

        # load fieldid2word list len 3
        self.fieldid2word = []
        with open(data_dir + "/field2word.txt") as f:
            for line in f:
                word_list = line.strip().split("\t")[1].split(" ")
                wordid_list = [int(tmp) for tmp in word_list]
                assert len(wordid_list) == 3
                self.fieldid2word.append(wordid_list)

        self.fieldid2word = np.array(self.fieldid2word)

        self.gpt_out_mask = []
        self.target_vocab = []
        with open(self.vocab_mask_path) as f:
            for line in f:
                line_list = line.strip().split()
                for ind, token in enumerate(line_list):
                    self.gpt_out_mask.append(int(token) + 0.0)
                    if token == "1":
                        self.target_vocab.append(ind)

        self.gpt_out_mask[-1] = 1.0
        if self.eos not in self.target_vocab:
            self.target_vocab.append(self.eos)
        if self.empty not in self.target_vocab:
            self.target_vocab.append(self.empty)

        assert len(self.gpt_out_mask) == eos + 1
        print(len(self.gpt_out_mask))
        print(len(self.target_vocab))

    def load_file(self, file_path):
        data = open(file_path).read().strip().split('\n')
        if self.limits > 0:
            data = data[:self.limits]
        print(len(data))
        print(data[0].strip().split(' '))
        d = [list(map(int, d.strip().split(' '))) for d in data]
        return d

    def load_data(self, split):
        """
        Load all data
        Args:
            split: str, one of 'train', 'test' or 'valid'

        Returns:

        """
        subdir = os.path.join(self.data_dir, split)
        file_path_suffixes = {'summary': '.summary.id',
                              'text': '.box.val.id',
                              'field': '.box.lab.id',
                              'pos': '.box.pos',
                              'rpos': '.box.rpos',
                              'dec': '_summary_field_id.txt',
                              'dec_pos': '_summary_pos.txt',
                              'dec_rpos': '_summary_rpos.txt',
                              'cont_path': '.context'}

        all_data = {}
        for fp in file_path_suffixes.keys():
            file_path = os.path.join(subdir, split + file_path_suffixes[fp])
            all_data[fp] = self.load_file(file_path)

        return all_data

    def shuffle_all_data(self, data):
        data_size = len(data['summary'])
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = {}
        for fp in data.keys():
            shuffled_data[fp] = np.array(data[fp])[shuffle_indices]
        return shuffled_data

    def get_zipped_batch(self, data, start_index, end_index):
        return zip(data['summary'][start_index:end_index],
                   data['text'][start_index:end_index],
                   data['field'][start_index:end_index],
                   data['pos'][start_index:end_index],
                   data['rpos'][start_index:end_index],
                   data['dec'][start_index:end_index],
                   data['dec_pos'][start_index:end_index],
                   data['dec_rpos'][start_index:end_index],
                   data['cont_path'][start_index:end_index])


    def batch_iter(self, data, batch_size, shuffle, domain):
        #summaries, texts, fields, poses, rposes, decoder_field, decoder_pos, decoder_rpos, context_in = data
        data_size = len(data['summary'])
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            data = self.shuffle_all_data(data)

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            max_summary_len = max([len(sample) for sample in data['summary'][start_index:end_index]])
            max_text_len = max([len(sample) for sample in data['text'][start_index:end_index]])
            max_cont_len = max([len(sample) for sample in data['cont_path'][start_index:end_index]])

            batch_data = {'enc_in': [], 'enc_fd': [], 'enc_pos': [], 'enc_rpos': [], 'enc_len': [],
                          'dec_in': [], 'dec_len': [], 'dec_out': [], 'oov_map': [],
                          'dec_field': [], 'dec_pos': [], 'dec_rpos': [], 'gpt_context': [],
                          'context': [], 'enc_in_real': [], 'dec_in_real': []}

            data_subset = self.get_zipped_batch(data, start_index, end_index)

            for summary, text, field, pos, rpos, dec_field, dec_pos, dec_rpos, cont_text in data_subset:
                summary_len = len(summary)
                text_len = len(text)
                cont_len = len(cont_text)
                pos_len = len(pos)
                rpos_len = len(rpos)
                assert text_len == len(field)
                assert pos_len == len(field)
                assert rpos_len == pos_len

                assert len(dec_field) == len(summary)

                gold = summary + [self.eos] * (max_summary_len - summary_len + 1)
                # context = [self.eos] * (max_summary_len - summary_len) + summary
                summary = summary + [self.eos] * (max_summary_len - summary_len)

                dec_field = dec_field + [self.empty] * (max_summary_len - summary_len)
                dec_pos = dec_pos + [0] * (max_summary_len - summary_len)
                dec_rpos = dec_rpos + [0] * (max_summary_len - summary_len)

                context = [self.empty] * (max_cont_len - cont_len) + cont_text

                text = text + [self.empty] * (max_text_len - text_len)
                field = field + [self.empty] * (max_text_len - text_len)
                pos = pos + [0] * (max_text_len - text_len)
                rpos = rpos + [0] * (max_text_len - text_len)

                if max_text_len > self.man_text_len:
                    text = text[:self.man_text_len]

                    context = context[-self.man_text_len:]

                    field = field[:self.man_text_len]
                    pos = pos[:self.man_text_len]
                    rpos = rpos[:self.man_text_len]
                    text_len = min(text_len, self.man_text_len)

                elif max_cont_len > self.man_text_len:
                    context = context[-self.man_text_len:]

                # OOM
                if max_summary_len > self.man_summary_len:
                    gold = gold[:self.man_summary_len + 1]
                    summary = summary[:self.man_summary_len]

                    # context = context[-self.man_summary_len:]

                    dec_field = dec_field[:self.man_summary_len]
                    dec_pos = dec_pos[:self.man_summary_len]
                    dec_rpos = dec_rpos[:self.man_summary_len]
                    summary_len = min(summary_len, self.man_summary_len)

                gpt_context = None
                if domain == "humans":
                    gpt_context = " Biography : "
                elif domain == "books":
                    gpt_context = " Book introduction : "
                elif domain == "songs":
                    gpt_context = " Song introduction : "

                gpt_context, _ = enc.encode(gpt_context)

                # ### check:
                # for token in gold:
                #     if token not in self.target_vocab:
                #         print ("Error!")
                

                # vocab mask
                text_real = []
                for token in text:
                    if token in self.target_vocab:
                        text_real.append(token)
                    else:
                        text_real.append(self.empty)

                dec_real = []
                for token in summary:
                    if token in self.target_vocab:
                        dec_real.append(token)
                    else:
                        dec_real.append(self.empty)

                batch_data['enc_in'].append(text)
                batch_data['enc_len'].append(text_len)
                batch_data['enc_fd'].append(field)
                batch_data['enc_pos'].append(pos)
                batch_data['enc_rpos'].append(rpos)
                batch_data['dec_in'].append(summary)
                batch_data['dec_len'].append(summary_len)
                batch_data['dec_out'].append(gold)
                batch_data['dec_field'].append(dec_field)
                batch_data['dec_pos'].append(dec_pos)
                batch_data['dec_rpos'].append(dec_rpos)
                batch_data['gpt_context'].append(gpt_context)
                batch_data['context'].append(context)

                batch_data['enc_in_real'].append(text_real)
                batch_data['dec_in_real'].append(dec_real)

            yield batch_data


