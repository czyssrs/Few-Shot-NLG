from evaluate_inferece import extract_structure_from_sample, parse_file, COMPARISONS
from tqdm import tqdm
import sys
from preprocess import join_box
import numpy as np

if __name__ == "__main__":
    file_name = sys.argv[1]
    dem_file = sys.argv[2]
    elements = parse_file(file_name)
    n_fields = 50

    facts_file = "amt_facts_data.csv"
    facts_key_file = "amt_facts_keys.csv"
    with open(facts_file, "w") as fp:
        with open(facts_key_file, "w") as fp_key:
            # header for facts file
            header_str = "summary,"
            header_str += ",".join(["field"+str(x+1)+","+"value"+str(x+1) for x in range(n_fields)])
            fp.write(header_str + "\n")
            n_f1 = len(header_str.split(","))

            # header for key file
            header_str = "key" + "\n"
            fp_key.write(header_str)

            for sample in tqdm(elements):
                results = extract_structure_from_sample(sample)
                if results is not None:
                    box = results['box']
                    box_list = box.strip().split("\t")
                    box_out_list, _ = join_box(box_list)
                    n_sample = len(box_out_list)
                    value_str = ""
                    for (k, v) in box_out_list:
                        k = "\"" + k + "\"" #FIXME
                        v = "\"" + v + "\""
                        value_str += k+","+v+","

                    empty_fields = 2*(n_fields - n_sample) - 1
                    for c in COMPARISONS:
                        value_str1 = "\"" + results[c] + "\"" + "," + value_str + "".join(
                            empty_fields * [","]) + '\n'
                        fp.write(value_str1)
                        fp_key.write(c + '\n')

    comparisons_file = "amt_grammar_comparisons.csv"
    comparisons_key_file = "amt_grammar_comparisons_keys.csv"

    with open(comparisons_file, "w") as fp:
        with open(comparisons_key_file, "w") as fp_key:
            # header for facts file
            header_str = "text1,text2"
            fp.write(header_str + "\n")

            # header for key file
            header_str = "key1,key2" + "\n"
            fp_key.write(header_str)

            for sample in tqdm(elements):
                results = extract_structure_from_sample(sample)
                if results is not None:
                    n = len(COMPARISONS)
                    for e1 in range(n):
                        for e2 in range(e1+1, n):
                            if np.random.rand(1)[0] > 0.5:
                                x1, x2 = e1, e2
                            else:
                                x1, x2 = e2, e1
                            value_str1 = "\"" + results[COMPARISONS[x1]] + "\"" + "," + "\"" + \
                                         results[COMPARISONS[x2]] + "\"" + "\n"
                            fp.write(value_str1)
                            fp_key.write(COMPARISONS[x1] + "," + COMPARISONS[x2] + '\n')






    import ipdb; ipdb.set_trace()