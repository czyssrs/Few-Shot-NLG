import re, time, os

# root_path = "/scratch/home/zhiyu/wiki2bio/"
root_path = "../emb_baseline_pointer/"
# merge_word_vocab = "crawled_data/merged_vocab.txt"

merge_field_vocab = root_path + "human_books_songs_films_field_vocab.txt"
word_vocab = root_path + "human_books_songs_films_word_vocab_2000.txt"

### vocab 2000 max 30
extend_vocab_size = 30


def split_infobox():
    """
    extract box content, field type and position information from infoboxes from original_data
    *.box.val is the box content (token)
    *.box.lab is the field type for each token
    *.box.pos is the position counted from the begining of a field
    """
    bwfile = [root_path + "processed_data/train/train.box.val", 
              root_path + "processed_data/valid/valid.box.val", 
              root_path + "processed_data/test/test.box.val"]
    bffile = [root_path + "processed_data/train/train.box.lab", 
              root_path + "processed_data/valid/valid.box.lab", 
              root_path + "processed_data/test/test.box.lab"]
    bpfile = [root_path + "processed_data/train/train.box.pos", 
              root_path + "processed_data/valid/valid.box.pos", 
              root_path + "processed_data/test/test.box.pos"]
    boxes = [root_path + "original_data/train.box", root_path + "original_data/valid.box", root_path + "original_data/test.box"]
    
    mixb_word, mixb_label, mixb_pos = [], [], []
    for fboxes in boxes:
        box = open(fboxes, "r").read().strip().split('\n')
        box_word, box_label, box_pos = [], [], []
        for ib in box:
            item = ib.split('\t')
            box_single_word, box_single_label, box_single_pos = [], [], []
            for it in item:
                if len(it.split(':')) > 2:
                    continue
                # print it
                prefix, word = it.split(':')
                if '<none>' in word or word.strip()=='' or prefix.strip()=='':
                    continue
                new_label = re.sub("_[1-9]\d*$", "", prefix)
                if new_label.strip() == "":
                    continue
                box_single_word.append(word)
                box_single_label.append(new_label)
                if re.search("_[1-9]\d*$", prefix):
                    field_id = int(prefix.split('_')[-1])
                    box_single_pos.append(field_id if field_id<=30 else 30)
                else:
                    box_single_pos.append(1)
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


def reverse_pos():
    # get the position counted from the end of a field
    bpfile = [root_path + "processed_data/train/train.box.pos", root_path + "processed_data/valid/valid.box.pos", root_path + "processed_data/test/test.box.pos"]
    bwfile = [root_path + "processed_data/train/train.box.rpos", root_path + "processed_data/valid/valid.box.rpos", root_path + "processed_data/test/test.box.rpos"]
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

def check_generated_box():
    ftrain = [root_path + "processed_data/train/train.box.val",
              root_path + "processed_data/train/train.box.lab",
              root_path + "processed_data/train/train.box.pos",
              root_path + "processed_data/train/train.box.rpos"]
    ftest  = [root_path + "processed_data/test/test.box.val", 
              root_path + "processed_data/test/test.box.lab",
              root_path + "processed_data/test/test.box.pos",
              root_path + "processed_data/test/test.box.rpos"]
    fvalid = [root_path + "processed_data/valid/valid.box.val", 
              root_path + "processed_data/valid/valid.box.lab", 
              root_path + "processed_data/valid/valid.box.pos",
              root_path + "processed_data/valid/valid.box.rpos"]
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
                print case
                print val
                print len(vval)
                print len(llab)
                print len(ppos)
                print len(rrpos)
            assert len(vval) == len(llab)
            assert len(llab) == len(ppos)
            assert len(ppos) == len(rrpos)


def split_summary_for_rouge():
    bpfile = [root_path + "original_data/test.summary", root_path + "original_data/valid.summary"]
    bwfile = [root_path + "processed_data/test/test_split_for_rouge/", root_path + "processed_data/valid/valid_split_for_rouge/"]
    for i, fi in enumerate(bpfile):
        fread = open(fi, 'r')
        k = 0
        for line in fread:
            with open(bwfile[i] + 'gold_summary_' + str(k), 'w') as sw:
                sw.write(line.strip() + '\n')
            k += 1
        fread.close()



class Vocab(object):
    """vocabulary for words and field types"""
    def __init__(self):
        vocab = dict()
        vocab['<_PAD>'] = 0
        vocab['<_START_TOKEN>'] = 1
        vocab['<_END_TOKEN>'] = 2
        vocab['<_UNK_TOKEN>'] = 3
        cnt = 4
        # with open(root_path + "original_data/word_vocab.txt", "r") as v:
        with open(word_vocab, "r") as v:
            for line in v:
                if len(line.strip().split()) > 1:
                    word = line.strip().split()[0]
                    ori_id = int(line.strip().split()[1])
                    if word not in vocab:
                        vocab[word] = (cnt + ori_id)

        self._word2id = vocab
        self._id2word = {value: key for key, value in vocab.items()}
        print len(vocab)
        self.vocab_size = len(vocab)

        key_map = dict()
        key_map['<_PAD>'] = 0
        key_map['<_START_TOKEN>'] = 1
        key_map['<_END_TOKEN>'] = 2
        key_map['<_UNK_TOKEN>'] = 3
        cnt = 4
        # with open(root_path + "original_data/field_vocab.txt", "r") as v:
        with open(merge_field_vocab, "r") as v:
            for line in v:
                key = line.strip().split()[0]
                key_map[key] = cnt
                cnt += 1
        self._key2id = key_map
        self._id2key = {value: key for key, value in key_map.items()}
        print len(key_map)

        #### add for field id to word group mapping
        self._keyid2wordlist =  dict()
        self._keyid2wordlist[0] = [0,0,0]
        self._keyid2wordlist[1] = [0,0,0]
        self._keyid2wordlist[2] = [0,0,0]
        self._keyid2wordlist[3] = [0,0,0]

        for i in range(4, len(self._id2key)):
            # self._keyid2wordlist[i] = [self._word2id[tmp] for tmp in self._id2key[i].split("_")]
            self._keyid2wordlist[i] = []
            for tmp in self._id2key[i].split("_"):
                if tmp in self._word2id:
                    self._keyid2wordlist[i].append(self._word2id[tmp])
                else:
                    self._keyid2wordlist[i].append(3)

            if len(self._keyid2wordlist[i]) > 3:
                self._keyid2wordlist[i] = self._keyid2wordlist[i][:3]
            else:
                extended = 3 - len(self._keyid2wordlist[i])
                self._keyid2wordlist[i] += ([0] * extended)

    def word2id(self, word):
        ans = self._word2id[word] if word in self._word2id else 3
        return ans

    def id2word(self, id):
        ans = self._id2word[int(id)]
        return ans

    def key2id(self, key):
        ans = self._key2id[key] if key in self._key2id else 3
        return ans

    def id2key(self, id):
        ans = self._id2key[int(id)]
        return ans

def table2id():
    fvals = [root_path + 'processed_data/train/train.box.val',
             root_path + 'processed_data/test/test.box.val',
             root_path + 'processed_data/valid/valid.box.val']
    flabs = [root_path + 'processed_data/train/train.box.lab',
             root_path + 'processed_data/test/test.box.lab',
             root_path + 'processed_data/valid/valid.box.lab']
    fsums = [root_path + 'original_data/train.summary',
             root_path + 'original_data/test.summary',
             root_path + 'original_data/valid.summary']

    fvals2id = [root_path + 'processed_data/train/train.box.val.id',
                root_path + 'processed_data/test/test.box.val.id',
                root_path + 'processed_data/valid/valid.box.val.id']
    flabs2id = [root_path + 'processed_data/train/train.box.lab.id',
                root_path + 'processed_data/test/test.box.lab.id',
                root_path + 'processed_data/valid/valid.box.lab.id']
    fsums2id = [root_path + 'processed_data/train/train.summary.id',
                root_path + 'processed_data/test/test.summary.id',
                root_path + 'processed_data/valid/valid.summary.id']

    f_local_vocab = [root_path + 'processed_data/train/train_local_oov.txt',
                    root_path + 'processed_data/test/test_local_oov.txt',
                    root_path + 'processed_data/valid/valid_local_oov.txt']

    # f_decoder_pos = [root_path + 'processed_data/train/train_summary_pos.txt',
    #                 root_path + 'processed_data/test/test_lsummary_pos.txt',
    #                 root_path + 'processed_data/valid/valid_summary_pos.txt']

    # f_decoder_rpos = f_decoder_pos = [root_path + 'processed_data/train/train_summary_rpos.txt',
    #                                     root_path + 'processed_data/test/test_lsummary_rpos.txt',
    #                                     root_path + 'processed_data/valid/valid_summary_rpos.txt']



    vocab = Vocab()
    vocab_size = vocab.vocab_size

    ### field not change
    for k, ff in enumerate(flabs):
        fi = open(ff, 'r')
        fo = open(flabs2id[k], 'w')
        for line in fi:
            items = line.strip().split()
            fo.write(" ".join([str(vocab.key2id(key)) for key in items]) + '\n')
        fi.close()
        fo.close()

    ### write field to word mapping
    field2word_file = root_path + "processed_data/field2word.txt"
    with open(field2word_file, "w") as f:
        for each_id in vocab._keyid2wordlist:
            f.write(str(each_id) + "\t" + " ".join([str(tmp) for tmp in vocab._keyid2wordlist[each_id]]) + "\n")



    # ### val and sum extend vocab
    # for k, ff in enumerate(fsums):
    #     fi = open(ff, 'r')
    #     fo = open(fsums2id[k], 'w')
    #     for line in fi:
    #         items = line.strip().split()
    #         fo.write(" ".join([str(vocab.word2id(word)) for word in items]) + '\n')
    #     fi.close()
    #     fo.close()

    # for k, ff in enumerate(fvals):
    #     fi = open(ff, 'r')
    #     fo = open(fvals2id[k], 'w')
    #     for line in fi:
    #         items = line.strip().split()
    #         fo.write(" ".join([str(vocab.word2id(word)) for word in items]) + '\n')
    #     fi.close()
    #     fo.close()

    f_test = root_path + "test_case.txt"
    f_t = open(f_test, "w")

    for k, (fs, fv) in enumerate(zip(fsums, fvals)):
        fsum = open(fs)
        fsumo = open(fsums2id[k], 'w')

        fval = open(fv)
        fvalo = open(fvals2id[k], 'w')

        f_oov = open(f_local_vocab[k], "w")

        lines_sum = fsum.readlines()
        lines_val = fval.readlines()

        all_oov = 0
        find_in_source = 0
        max_oov = 0
        max_sub = 0

        for line_sum, line_val in zip(lines_sum, lines_val):
            line_sum_list = line_sum.strip().split()
            line_val_list = line_val.strip().split()

            local_oov = {}
            num_local_oov = 0

            res_sum_list = []
            res_val_list = []

            for token in line_sum_list:
                this_vocab_id = vocab.word2id(token)
                if this_vocab_id != 3:
                    res_sum_list.append(this_vocab_id)
                else:
                    if num_local_oov > extend_vocab_size:
                        res_sum_list.append(this_vocab_id)
                        continue
                    ## oov
                    # in val oov
                    if (token in line_val_list) and (token not in local_oov):
                        local_oov[token] = (vocab_size + num_local_oov)
                        num_local_oov += 1

                        res_sum_list.append(local_oov[token])

            for token in line_val_list:
                this_vocab_id = vocab.word2id(token)
                if this_vocab_id != 3:
                    res_val_list.append(this_vocab_id)
                else:
                    if token in local_oov:
                        res_val_list.append(local_oov[token])
                    else:
                        res_val_list.append(this_vocab_id)


            fsumo.write(" ".join([str(tmp) for tmp in res_sum_list]) + "\n")
            fvalo.write(" ".join([str(tmp) for tmp in res_val_list]) + "\n")


            # print "\n"
            # print " ".join([str(tmp) for tmp in res_sum_list])
            # print " ".join([str(tmp) for tmp in line_sum_list])
            # print " ".join([str(tmp) for tmp in res_val_list])
            # print " ".join([str(tmp) for tmp in line_val_list])

            f_t.write(" ".join([str(tmp) for tmp in res_sum_list]) + "\n")
            f_t.write(" ".join([str(tmp) for tmp in line_sum_list]) + "\n")
            f_t.write(" ".join([str(tmp) for tmp in res_val_list]) + "\n")
            f_t.write(" ".join([str(tmp) for tmp in line_val_list]) + "\n")
            f_t.write("\n")


            all_oov += len(local_oov)
            if max_oov < len(local_oov):
                max_oov = len(local_oov)


            oov_write = "\t".join([str(local_oov[token]) + ":" + token for token in local_oov])
            f_oov.write(oov_write + "\n")


        fsumo.close()
        fvalo.close()
        f_oov.close()


        print "Avg oov: ", float(all_oov) / len(lines_sum)
        print "Max oov: ", max_oov

    f_t.close()


def preprocess():
    """
    We use a triple <f, p+, p-> to represent the field information of a token in the specific field. 
    p+&p- are the position of the token in that field counted from the begining and the end of the field.
    For example, for a field (birthname, Jurgis Mikelatitis) in an infoboxes, we represent the field as
    (Jurgis, <birthname, 1, 2>) & (Mikelatitis, <birthname, 2, 1>)
    """
    # print("extracting token, field type and position info from original data ...")
    # time_start = time.time()
    # split_infobox()
    # reverse_pos()
    # duration = time.time() - time_start
    # print("extract finished in %.3f seconds" % float(duration))

    # print("spliting test and valid summaries for ROUGE evaluation ...")
    # time_start = time.time()
    # split_summary_for_rouge()
    # duration = time.time() - time_start
    # print("split finished in %.3f seconds" % float(duration))

    print("turning words and field types to ids ...")
    time_start = time.time()
    table2id()
    duration = time.time() - time_start
    print("idlization finished in %.3f seconds" % float(duration))



def make_dirs():
    os.mkdir(root_path + "results/")
    os.mkdir(root_path + "results/res/")
    os.mkdir(root_path + "results/evaluation/")
    os.mkdir(root_path + "processed_data/")
    os.mkdir(root_path + "processed_data/train/")
    os.mkdir(root_path + "processed_data/test/")
    os.mkdir(root_path + "processed_data/valid/")
    os.mkdir(root_path + "processed_data/test/test_split_for_rouge/")
    os.mkdir(root_path + "processed_data/valid/valid_split_for_rouge/")

if __name__ == '__main__':
    # make_dirs()
    preprocess()
    check_generated_box()
    print("check done")













    
