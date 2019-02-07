import sys
import os
import json
import urllib2
import re
import corenlp
import time
from tqdm import tqdm
from BeautifulSoup import BeautifulSoup, NavigableString, Tag

raw_dest_path = "/scratch/home/zhiyu/wiki2bio/crawled_data/raw/films/"
wiki_root = "https://en.wikipedia.org"


out_box = "/scratch/home/zhiyu/wiki2bio/crawled_data/films.box"
out_summary = "/scratch/home/zhiyu/wiki2bio/crawled_data/films.summary"

out_raw = "/scratch/home/zhiyu/wiki2bio/crawled_data/films.raw.txt"


############
'''
final choice : books, songs, films
'''
############



def get_all_pages(root, dest_path):
	'''
	download all wiki pages from root category
	'''
	try:
		page = urllib2.urlopen(root)
		soup = BeautifulSoup(page)
	except:
		return 0

	num_retrieved = 0


	### subcategory recursion
	x = soup.body.find('div', attrs={'id' : 'mw-subcategories'})

	if x != None:
		for item in x.contents:
			if isinstance(item, NavigableString):
				continue
			if isinstance(item, Tag):
				try:
					tmp = item.findAll('a')
					### first one exclude
					if len(tmp) > 1:
						for each_tmp in tmp[1:]:
							next_root = wiki_root + each_tmp.get('href')
							num_retrieved += get_all_pages(next_root, dest_path)

				except:
					continue




	### retrieve pages
	x = soup.body.find('div', attrs={'id' : 'mw-pages'})

	if x != None:
		for item in x.contents:
			if isinstance(item, NavigableString):
				continue
			if isinstance(item, Tag):
				tmp = item.findAll('a')
				### first one exclude
				if len(tmp) > 1:
					for each_tmp in tmp[1:]:

						try:
							# print each_tmp.get('href')
							name = each_tmp.get('href').strip("/").replace("/", "_")
							this_page_url = wiki_root + each_tmp.get('href')

							this_page_content = urllib2.urlopen(this_page_url)

							with open(dest_path + name + ".html", "w") as f:
								f.write(this_page_content.read())

							num_retrieved += 1

						except:
							continue

	return num_retrieved





def gen_data(source_path, out_file):
	'''
	generate data as table2text format
	'''

	# url_test = source_path + "_wiki_Amiga_500"
	# soup = BeautifulSoup(open(url_test), "html.parser")

	num_recorded = 0
	num_files = 0
	output_file = open(out_file, "w")

	for (dirpath, dirnames, filenames) in os.walk(raw_dest_path):
		for file in tqdm(filenames):

			try:
				num_files += 1
				file_in = source_path + file
				this_name = file.strip("wiki_").strip(".html")

				# file_in = source_path + "wiki_Amiga_500.html"
				### filters
				# for films
				if "filmography" in this_name.lower():
					# print "skip"
					continue

				# for song
				if "list" in this_name.lower():
					continue


				page = open(file_in)
				soup = BeautifulSoup(page.read())

				### <table class="infobox hproduct vevent" 
				if soup.body == None:
					continue

				x = soup.body.find('table', attrs={'class' : re.compile('infobox?[a-z0-9]')})

				# ### <b>Amiga 500</b>
				# entity_name = file_in.split("wiki_")[1].strip(".html").replace("_", " ")
				# bio_keyword = "<b>" + entity_name + "</b>"

				# print bio_keyword

				y = soup.body.findAll('p')

				if x != None and len(y) > 0:

					### for summary. min valid length
					i = 0
					while i < len(y) and len(text_cleaner(y[i])) < 20:
						i += 1
					if i == len(y):
						continue

					cleanbio = text_cleaner(y[i])
					cleanbio_list = cleanbio.split(". ")
					cleanbio = cleanbio_list[0]

					i = 1
					while cleanbio[-2] == " ":
						cleanbio += (". " + cleanbio_list[i])
						i += 1

					if len(cleanbio) < 20:
						continue

					infobox_list = []
					infobox_list.append("name:" + this_name)

					### for infobox
					for item in x.contents:
						if isinstance(item, NavigableString):
							continue
						if isinstance(item, Tag):
							tmp = item.findAll('tr')
							if len(tmp) > 0:
								for each_row in tmp:
									# print each_row
									field = each_row.find('th')
									value = each_row.find('td')
									# if field != None:
									# 	print text_cleaner(field) + "\t" + text_cleaner(value)

									if field != None and value != None:
										infobox_list.append(text_cleaner(field).replace(" ", "_").replace(":", "_") + ":" + text_cleaner(value).replace(" ", "_").replace(":", "_"))

					if len(infobox_list) > 1:
						num_recorded += 1
						output_file.write(" ".join(infobox_list).encode('utf-8') + "\t")
						output_file.write(cleanbio.encode('utf-8').replace("\t", " ").replace("\n", " ") + "\n")

			except:
				continue


	output_file.close()

	print "All files: ", num_files
	print "Recorded: ", num_recorded



def text_cleaner(tag_in):

	res_bio = ""
	for token in tag_in.contents:

		if isinstance(token, NavigableString):
			res_bio += token.string
		if isinstance(token, Tag):
			res_bio += token.text

	cleanr = re.compile('&#?[a-z0-9]{2,8};')
	cleanbio = re.sub(cleanr, ' ', res_bio)
	cleanbio = re.sub('\s+', ' ', cleanbio).strip()

	return cleanbio



def tokenize_and_filter(file_in, out_box, out_summary):
	'''
	stanford core nlp tokenizer;
	set of filters: summary must include info in infobox
	all lower cased
	original not include "."
	'''

	client = corenlp.client.CoreNLPClient(annotators="tokenize ssplit".split())

	# text = "I help a (test) ."
	# ann = client.annotate(text)
	# sentence = ann.sentence[0]
	# print [token.word for token in sentence.token]

	output_box = open(out_box, "w")
	output_summary = open(out_summary, "w")

	num_write = 0

	with open(file_in) as f:
		for line in tqdm(f.readlines()):
			# print line
			try:
				line = line.decode('utf-8')
				line_list = line.strip("\n").split("\t")
				field_list = line_list[0].strip().split(" ")
				summary = line_list[1]

				if summary[-1] != ".":
					summary += " ."

				### tokenize summary
				ann = client.annotate(summary)
				sentence = ann.sentence[0]
				summary = " ".join([token.word for token in sentence.token]).lower()

				field_value_emit_list = []

				invalid_flag = 0
				valid_flag = 0

				for each_field in field_list:

					if ":" not in each_field:
						## bad field
						continue
					field_name = each_field.split(":")[0]
					field_value = each_field.split(":")[1]

					if field_name and field_value:
						field_name = field_name.replace("(s)", "s")
						field_name = field_name.replace("_", " ")
						## tokenize field name
						ann = client.annotate(field_name)
						sentence = ann.sentence[0]
						field_name = "_".join([token.word for token in sentence.token]).lower()

						##### constraints list here: not born! 
						if field_name == "born":
							invalid_flag = 1


						field_value = field_value.replace("(s)", "s")
						field_value = field_value.replace("_", " ")
						## tokenize field name
						ann = client.annotate(field_value)
						sentence = ann.sentence[0]
						field_value = " ".join([token.word for token in sentence.token]).lower()

						if field_value in summary:
							valid_flag = 1

						fv_list = field_value.split(" ")

						for ind, fv_seg in enumerate(fv_list):

							emit_seg = field_name + "_" + str(ind + 1) + ":" + fv_seg
							field_value_emit_list.append(emit_seg)


				### threshold here?
				if len(field_value_emit_list) > 1 and invalid_flag == 0:
					if valid_flag == 1:

						# print "\t".join(field_value_emit_list)
						# print summary

						output_box.write("\t".join(field_value_emit_list).encode('utf-8') + "\n")
						output_summary.write(summary.encode('utf-8') + "\n")
						num_write += 1

			except:

				time.sleep(10)




	output_box.close()
	output_summary.close()

	print "Final write: ", num_write







if __name__=='__main__':

	#### download pages
	#root_page = "https://en.wikipedia.org/wiki/Category:Personal_computers"
	#root_page = "https://en.wikipedia.org/wiki/Category:Companies_by_country"
	#root_page = "https://en.wikipedia.org/wiki/Category:Songs"
	#root_page = "https://en.wikipedia.org/wiki/Category:Universities_and_colleges_by_country"
	#root_page = "https://en.wikipedia.org/wiki/Category:Books_by_country"
	# root_page = "https://en.wikipedia.org/wiki/Category:Films_by_country"
	# retrieved = get_all_pages(root_page, raw_dest_path)
	# print retrieved


	#### generate data: info box and summary
	# gen_data(raw_dest_path, out_raw)

	#### tokenize and other filters
	tokenize_and_filter(out_raw, out_box, out_summary)
























