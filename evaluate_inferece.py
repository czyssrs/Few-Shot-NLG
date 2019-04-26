import sys
from preprocess import join_box, load_dem_map, fuzzy_match_rep
from tqdm import tqdm

SKIP_FIELDS = ['caption', 'image', 'image_size', 'article_title', '']
COMPARISONS = ["gold", "gpt", "switch", "ours"]

def parse_file(full_path):
    with open(full_path, "r") as fp:
        elements = []
        new_element = []
        count = 0
        for line in fp:
            if line == "###################################\n":
                elements.append(new_element)
                new_element = []
                count += 1
                print(count)
            else:
                new_element.append(line)
    return elements


def extract_structure_from_sample(sample):
    if len(sample) == 1 and sample[0] == "\n":
        return None

    assert len(sample) == 2 + 2 * len(COMPARISONS)

    # line 0 is box
    infobox = sample[0]
    assert sum([1 for x in infobox if x == ":"]) > 0
    # line 1 is Gold
    assert sample[1] == "Gold: \n"
    gold = sample[2].strip()
    # line 3 is switch
    assert sample[3] == "Switch: \n"
    switch = sample[4].strip()
    # line 5 is GPT
    assert sample[5] == "GPT only: \n"
    gpt = sample[6].strip()
    # line 5 is ours
    assert sample[7] == "Ours: \n"
    ours = sample[8].strip()

    return {"box": infobox, "gold": gold, "gpt": gpt, "ours": ours, "switch": switch}


def replace_summary_with_field_pos(box, summary, dem_file):
    ### load nationality demonyms.csv
    dem_map = load_dem_map(dem_file)

    box = box.replace("-lrb-", "(")
    box = box.replace("-rrb-", ")")

    box_list = box.strip().split("\t")
    box_out_list, box_field_list = join_box(box_list)

    summary = summary.replace("-lrb-", "(")
    summary = summary.replace("-rrb-", ")")

    tem_summary = summary.strip()
    out_summary = summary.strip()
    tem_summary_list = tem_summary.split(" ")


    summarized_fields = []
    not_summarized_fields = []
    for (this_name, this_value) in box_field_list:
        # print(this_name)
        this_name = this_name.strip()
        this_value = this_value.strip()
        if this_name in SKIP_FIELDS:
            continue
        this_value_dict = {}

        for ind, each_token in enumerate(this_value.split(" ")):
            # if each_token not in this_value_dict:
            this_value_dict[each_token] = ind + 1

        this_value_list_len = len(this_value.split(" "))

        found = False
        if " " + this_value + " " in out_summary:
            out_summary = out_summary.replace(" " + this_value + " ", " " + ("<" + this_name + "> ") * this_value_list_len)
            found = True
            summarized_fields.append(this_name)
        # name
        elif out_summary.startswith(this_value + " "):
            out_summary = out_summary.replace(this_value + " ", ("<" + this_name + "> ") * this_value_list_len)
            found = True
            summarized_fields.append(this_name)
        # nationality
        elif this_value in dem_map:
            this_value_list = dem_map[this_value]
            for this_value in this_value_list:
                this_value_list_len = len(this_value.split(" "))
                if " " + this_value + " " in out_summary:
                    out_summary = out_summary.replace(" " + this_value + " ", " " + ("<" + this_name + "> ") * this_value_list_len)
                    found = True
                    summarized_fields.append(this_name)

        else:
            # seperate nationality
            is_dem_match = 0
            this_value_list = this_value.split(" , ")
            if len(this_value_list) > 1:
                for each_con in this_value_list:
                    if " " + each_con + " " in out_summary and each_con in dem_map:
                        each_con_len = len(each_con.split(" "))
                        out_summary = out_summary.replace(" " + each_con + " ", " " + ("<" + this_name + "> ") * each_con_len)
                        is_dem_match = 1
                        found = True
                        summarized_fields.append(this_name)
                        break
                    if each_con in dem_map:
                        this_con_list = dem_map[each_con]
                        for this_con in this_con_list:
                            if " " + this_con + " " in out_summary:
                                this_con_len = len(this_con.split(" "))
                                out_summary = out_summary.replace(" " + this_con + " ", " " + ("<" + this_name + "> ") * this_con_len)
                                is_dem_match = 1
                                found = True
                                summarized_fields.append(this_name)
                                break

            if is_dem_match:
                continue

            old_out_summary = out_summary[:]
            out_summary = fuzzy_match_rep(out_summary, this_value, this_name)
            if out_summary != old_out_summary:
                found = True
                summarized_fields.append(this_name)

        if not found and this_name in ["birth_date", "death_date"] and this_value != "unknown":
                this_value_year = this_value.strip().split(" ")[-1].strip().strip(")")
                try:
                    (int(this_value_year) > 0 and int(this_value_year) < 2019)
                except ValueError:
                    continue
                if " " + this_value_year + " " in out_summary:
                    out_summary = out_summary.replace(" " + this_value_year + " ", " " + ("<" + this_name + "> "))
                    found = True
                    summarized_fields.append(this_name)

        if not found:
            not_summarized_fields.append(this_name)

        assert len(out_summary.split(" ")) == len(tem_summary_list)

    assert len(out_summary.split(" ")) == len(tem_summary_list)

    # print(box)
    # print(summary)
    # print(out_summary)
    # import ipdb; ipdb.set_trace()
    return out_summary, summarized_fields, not_summarized_fields


if __name__ == "__main__":
    file_name = sys.argv[1]
    dem_file = sys.argv[2]
    elements = parse_file(file_name)
    avg_coverage = {'gold': [], 'gpt': [], 'ours': []}
    n_elem = 0
    for sample in tqdm(elements):
        results = extract_structure_from_sample(sample)
        if results is not None:
            n_elem += 1
            _, gold_coverage, gold_no_coverage = replace_summary_with_field_pos(results['box'],
                                                                                results['gold'],
                                                                                dem_file)
            _, gpt_coverage, gpt_no_coverage = replace_summary_with_field_pos(results['box'],
                                                                                results['gpt'],
                                                                                dem_file)
            _, ours_coverage, ours_no_coverage = replace_summary_with_field_pos(results['box'],
                                                                                results['ours'],
                                                                                dem_file)

            n_fields = len(gold_coverage) + len(gold_no_coverage)
            avg_coverage['gold'].append((1.* len(gold_coverage) / n_fields))
            avg_coverage['gpt'].append((1. * len(gpt_coverage) / n_fields))
            avg_coverage['ours'].append((1. * len(ours_coverage) / n_fields))

        import matplotlib.pyplot as plt

    colors = ['#E69F00', '#56B4E9', '#F0E442']
    plt.figure()
    plt.hist([avg_coverage['gold'], avg_coverage['gpt'], avg_coverage['ours']],
             bins=20, stacked=False, normed=True, color=colors,
             label=['gold', 'gpt', 'ours'])
    plt.legend()
    plt.savefig("facts.png")
    import ipdb; ipdb.set_trace()


