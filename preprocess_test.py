import re, time, os, sys

# root_path = "/scratch/home/zhiyu/wiki2bio/"
root_path = "../emb_baseline/other_domain_data/"
# merge_word_vocab = "crawled_data/merged_vocab.txt"

merge_field_vocab = "../emb_baseline/pc_books_songs_field_vocab.txt"
word_vocab = "../emb_baseline/pc_books_songs_word_vocab.txt"

def split_infobox(test_name):
    """
    extract box content, field type and position information from infoboxes from original_data
    *.box.val is the box content (token)
    *.box.lab is the field type for each token
    *.box.pos is the position counted from the begining of a field
    """
    bwfile = [root_path + test_name + "/processed_data/test.box.val"]
    bffile = [root_path + test_name + "/processed_data/test.box.lab"]
    bpfile = [root_path + test_name + "/processed_data/test.box.pos"]
    boxes = [root_path + test_name + "/original_data/" + test_name + ".box"]
    
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


def reverse_pos(test_name):
    # get the position counted from the end of a field
    bpfile = [root_path + test_name + "/processed_data/test.box.pos"]
    bwfile = [root_path + test_name + "/processed_data/test.box.rpos"]
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

def check_generated_box(test_name):
    ftest  = [root_path + test_name + "/processed_data/test.box.val", 
              root_path + test_name + "/processed_data/test.box.lab",
              root_path + test_name + "/processed_data/test.box.pos",
              root_path + test_name + "/processed_data/test.box.rpos"]
    case = ftest
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


def split_summary_for_rouge(test_name):
    bpfile = [root_path + test_name + "/original_data/" + test_name + ".summary"]
    bwfile = [root_path + test_name + "/processed_data/test_split_for_rouge/"]
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

        # #### add for field id to word group mapping
        # self._keyid2wordlist =  dict()
        # self._keyid2wordlist[0] = [0,0,0]
        # self._keyid2wordlist[1] = [0,0,0]
        # self._keyid2wordlist[2] = [0,0,0]
        # self._keyid2wordlist[3] = [0,0,0]

        # for i in range(4, len(self._id2key)):
        #     # self._keyid2wordlist[i] = [self._word2id[tmp] for tmp in self._id2key[i].split("_")]
        #     self._keyid2wordlist[i] = []
        #     for tmp in self._id2key[i].split("_"):
        #         if tmp in self._word2id:
        #             self._keyid2wordlist[i].append(self._word2id[tmp])
        #         else:
        #             self._keyid2wordlist[i].append(3)

        #     if len(self._keyid2wordlist[i]) > 3:
        #         self._keyid2wordlist[i] = self._keyid2wordlist[i][:3]
        #     else:
        #         extended = 3 - len(self._keyid2wordlist[i])
        #         self._keyid2wordlist[i] += ([0] * extended)

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

def table2id(test_name):
    fvals = [root_path + test_name + '/processed_data/test.box.val']
    flabs = [root_path + test_name + '/processed_data/test.box.lab']
    fsums = [root_path + test_name + "/original_data/" + test_name + ".summary"]
    fvals2id = [root_path + test_name + '/processed_data/test.box.val.id']
    flabs2id = [root_path + test_name + '/processed_data/test.box.lab.id']
    fsums2id = [root_path + test_name + '/processed_data/test.summary.id']
    vocab = Vocab()
    for k, ff in enumerate(fvals):
        fi = open(ff, 'r')
        fo = open(fvals2id[k], 'w')
        for line in fi:
            items = line.strip().split()
            fo.write(" ".join([str(vocab.word2id(word)) for word in items]) + '\n')
        fi.close()
        fo.close()
    for k, ff in enumerate(flabs):
        fi = open(ff, 'r')
        fo = open(flabs2id[k], 'w')
        for line in fi:
            items = line.strip().split()
            fo.write(" ".join([str(vocab.key2id(key)) for key in items]) + '\n')
        fi.close()
        fo.close()
    for k, ff in enumerate(fsums):
        fi = open(ff, 'r')
        fo = open(fsums2id[k], 'w')
        for line in fi:
            items = line.strip().split()
            fo.write(" ".join([str(vocab.word2id(word)) for word in items]) + '\n')
        fi.close()
        fo.close()

    # ### write field to word mapping
    # field2word_file = root_path + "processed_data/field2word.txt"
    # with open(field2word_file, "w") as f:
    #     for each_id in vocab._keyid2wordlist:
    #         f.write(str(each_id) + "\t" + " ".join([str(tmp) for tmp in vocab._keyid2wordlist[each_id]]) + "\n")


    # ### check
    # with open(field2word_file) as f:
    #     for line in f:
    #         fieldind = int(line.strip().split("\t")[0])
    #         wordind = [int(tmp) for tmp in line.strip().split("\t")[1].split(" ")]
    #         print vocab.id2key(fieldind)

    #         for item in wordind:
    #             print vocab.id2word(item)

    #         print "\n"

def preprocess(test_name):
    """
    We use a triple <f, p+, p-> to represent the field information of a token in the specific field. 
    p+&p- are the position of the token in that field counted from the begining and the end of the field.
    For example, for a field (birthname, Jurgis Mikelatitis) in an infoboxes, we represent the field as
    (Jurgis, <birthname, 1, 2>) & (Mikelatitis, <birthname, 2, 1>)
    """
    print("extracting token, field type and position info from original data ...")
    time_start = time.time()
    split_infobox(test_name)
    reverse_pos(test_name)
    duration = time.time() - time_start
    print("extract finished in %.3f seconds" % float(duration))

    print("spliting test and valid summaries for ROUGE evaluation ...")
    time_start = time.time()
    split_summary_for_rouge(test_name)
    duration = time.time() - time_start
    print("split finished in %.3f seconds" % float(duration))

    print("turning words and field types to ids ...")
    time_start = time.time()
    table2id(test_name)
    duration = time.time() - time_start
    print("idlization finished in %.3f seconds" % float(duration))



def make_dirs(test_name):
    os.mkdir(root_path + test_name + "/results/")
    os.mkdir(root_path + test_name + "/results/res/")
    os.mkdir(root_path + test_name + "/results/evaluation/")
    os.mkdir(root_path + test_name + "/processed_data/")
    os.mkdir(root_path + test_name + "/processed_data/test_split_for_rouge/")

if __name__ == '__main__':

    test_name = sys.argv[1]
    make_dirs(test_name)
    preprocess(test_name)
    check_generated_box(test_name)
    print("check done")














