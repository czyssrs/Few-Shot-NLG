#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:43
# @Author  : Tianyu Liu

import time
import numpy as np
import encoder
import os

enc = encoder.get_encoder("117M")


class Preprocessor:
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
        """
        Load file, limit to self.limits lines, convert to list of lists
        Args:
            file_path: str, file path

        Returns:
            List of lists of tokens
        """
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
            Dict of data
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


class DataLoader:
    def __init__(self, data, target_vocab, domain, batch_size=64, shuffle=True, man_text_len=150,
                 man_summary_len=85, eos=50256, empty=28920):
        """
        Main dataloader
        Args:
            data_dir: dict, all the data
            batch_size: int, batch size
            shuffle: bool, Whether to shuffle data
            domain: str, domain name
        """
        self.data = data
        self.target_vocab = target_vocab
        self.domain = domain
        self.batch_size = batch_size
        self.man_text_len = man_text_len
        self.man_summary_len = man_summary_len
        self.eos = eos
        self.empty = empty
        self.data_size = len(data['summary'])
        self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
            else int(self.data_size / batch_size) + 1
        if shuffle:
            self.shuffle_all_data()
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        """
        Shuffle all data
        Returns:
            None
        """
        data_size = len(self.data['summary'])
        shuffle_indices = np.random.permutation(np.arange(data_size))
        for fp in self.data.keys():
            self.data[fp] = np.array(self.data[fp])[shuffle_indices]
        return

    def get_zipped_batch(self, data, start_index, end_index):
        """
        Get zipped batch of data given start and end index
        Args:
            data: Dict of data
            start_index: int, start index
            end_index: int, end index

        Returns:
            Iterable of batch data
        """
        return zip(data['summary'][start_index:end_index],
                   data['text'][start_index:end_index],
                   data['field'][start_index:end_index],
                   data['pos'][start_index:end_index],
                   data['rpos'][start_index:end_index],
                   data['dec'][start_index:end_index],
                   data['dec_pos'][start_index:end_index],
                   data['dec_rpos'][start_index:end_index],
                   data['cont_path'][start_index:end_index])

    def get_batch(self):
        if self.count >= self.num_batches:
            raise StopIteration("Ran out of data.")

        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        max_summary_len = max([len(sample) for sample in self.data['summary'][start_index:end_index]])
        max_text_len = max([len(sample) for sample in self.data['text'][start_index:end_index]])
        max_cont_len = max([len(sample) for sample in self.data['cont_path'][start_index:end_index]])

        batch_data = {'enc_in': [], 'enc_fd': [], 'enc_pos': [], 'enc_rpos': [], 'enc_len': [],
                      'dec_in': [], 'dec_len': [], 'dec_out': [], 'oov_map': [], 'dec_field': [],
                      'dec_pos': [], 'dec_rpos': [], 'gpt_context': [], 'context': [],
                      'enc_in_real': [], 'dec_in_real': []}

        data_subset = self.get_zipped_batch(self.data, start_index, end_index)

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
            if self.domain == "humans":
                gpt_context = " Biography : "
            elif self.domain == "books":
                gpt_context = " Book introduction : "
            elif self.domain == "songs":
                gpt_context = " Song introduction : "

            gpt_context, _ = enc.encode(gpt_context)

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

            batch_data['enc_in'].append(text)  # value
            batch_data['enc_len'].append(text_len)  # value length
            batch_data['enc_fd'].append(field)  # field
            batch_data['enc_pos'].append(pos)  # field p+
            batch_data['enc_rpos'].append(rpos)  # field p-
            batch_data['dec_in'].append(summary)  # summary
            batch_data['dec_len'].append(summary_len)  # summary len
            batch_data['dec_out'].append(gold)  # padded summary
            batch_data['dec_field'].append(dec_field)  # masked summary
            batch_data['dec_pos'].append(dec_pos)  # summary pos
            batch_data['dec_rpos'].append(dec_rpos)  # summary rpos
            batch_data['gpt_context'].append(gpt_context)  # box for gpt input with domain name
            batch_data['context'].append(context)  # padded context

            batch_data['enc_in_real'].append(text_real)
            batch_data['dec_in_real'].append(dec_real)

        return batch_data


