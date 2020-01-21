import gzip
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

MAX_LEN_IN_WORDS = 20
MAX_SAMPLES = 20000
#LANG_LIST = "en, fr"
LANG_LIST = "bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga"
LANG_LIST = [lang.strip() for lang in LANG_LIST.split(",")]
print(LANG_LIST)

def extract_data(dir_path, output_dir_path):

	onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	cache = {}

	for lang in LANG_LIST:
		cache[lang] = []

	for file_num in tqdm(range(0, len(onlyfiles))):
		only_file = onlyfiles[file_num]
		if '.gz' in only_file and 'incubator' not in only_file and 'simple' not in only_file:
			cache_text(dir_path, only_file, cache)

	populate_train_val_test(cache, output_dir_path)

def cache_text(dir_path, file_name, cache):
	
	source_language = file_name.split('.')[0]
	if source_language not in cache or len(cache[source_language])>=MAX_SAMPLES:
		return
	else:
		f = gzip.open(join(dir_path, file_name), 'rb')
		file_content = f.read()
		f.close()

		lines = file_content.split(b'\n')
		for line in lines:
			if len(line)==0:
				continue
			words = line.split(b' ')

			start = 0

			while start<len(words) and len(cache[source_language])<MAX_SAMPLES:
				st = ' '.encode().join(words[start:start+MAX_LEN_IN_WORDS])
				cache[source_language].append(st)
				start += MAX_LEN_IN_WORDS

def populate_train_val_test(cache, output_dir_path):
	val_proportion = 0.1
	test_proportion = 0.1
	
	f_train = open(join(output_dir_path, str(len(LANG_LIST))+'_train.tsv'), 'wb')
	f_val = open(join(output_dir_path, str(len(LANG_LIST))+'_val.tsv'), 'wb')
	f_test = open(join(output_dir_path, str(len(LANG_LIST))+'_test.tsv'), 'wb')

	header = b"text\tlanguage\n"
	f_train.write(header)
	f_val.write(header)
	f_test.write(header)

	for language in cache:
		sample_count = len(cache[language])
		sample_in_test = int(sample_count * test_proportion)
		sample_in_val = int(sample_count * test_proportion)
		sample_in_train = sample_count - sample_in_test - sample_in_val
		
		print('writing '+language)
		for counter in tqdm(range(0, sample_count)):
			st = cache[language][counter]+'\t'.encode()+language.encode()+'\n'.encode()
			if counter < sample_in_train:
				f_train.write(st)
			elif counter < (sample_in_train + sample_in_val):
				f_val.write(st)
			else:
				f_test.write(st)

	f_train.close()
	f_val.close()
	f_test.close()

if __name__=='__main__':
	dir_path = 'data'
	output_dir_path = 'data/extracted'

	extract_data(dir_path, output_dir_path)
