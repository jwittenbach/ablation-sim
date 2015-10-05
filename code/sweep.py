import os, sys, json, subprocess
from itertools import product
from copy import deepcopy
import numpy as np

def sliceToList(s):
	if len(s) == 1:
		start = 0
		stop = s[0]
		step = 1
	if len(s) == 2:
		start = s[0]
		stop = s[1]
		step = 1
	if len(s) == 3:
		start = s[0]
		stop = s[1]
		step = s[2]
	return np.arange(start, stop, step).tolist()

def getLists(d, l=[]):
	locs = []
	vals = []
	for k in d:
		data = d[k]
		new_path = l+[k]
		if type(data) is list:
			locs += [new_path]
			if data[0] == 'slice':
				vals += [sliceToList(data[1:])]
			else:
				vals += [data]
		elif type(data) is dict:
			new_lists, new_data = getLists(data, new_path)
			for n, dat in zip(new_lists, new_data):
				locs += [n]
				vals += [dat]
	return locs, vals

def setDict(d, k, v):
	'side effect warning: changes the passed in dictionary'
	ptr = d
	for l in k[:-1]:
		ptr = ptr[l]
	ptr[k[-1]] = v

# get parameters from config file
fname = sys.argv[2]
with open(fname, 'r') as f:
	params = json.loads(f.read())

print params
locs, vals = getLists(params)

inds, l_del, v_del = [], [], []

for (l, v) in zip(locs, vals):
	if v[0] == 'yoke':
		for (idx, match) in enumerate(locs):
			if match[-1] == v[1]:
				inds.append(idx)
				l_del.append(l)
				v_del.append(v)
				break

locs = [(l,) for l in locs]
for (idx, l, v) in zip(inds, l_del, v_del):
	locs[idx] += (l,)
	locs.remove((l,))
	vals.remove(v)

print '--------------'
print locs
print vals


output_location = os.path.abspath(params["simulation"]["result_location"])
if output_location[-1] != '/':
	output_location += '/'
 
inds = [range(len(v)) for v in vals]
labeled_vals = [zip(i, v) for (i, v) in zip(inds, vals)]

param_files = []
for p in product(*labeled_vals):
	
	labels, values = zip(* p)

	suffix = '-'.join([str(l) for l in labels])
	param_files.append('.params-' + suffix + ".json")
	
	d = deepcopy(params)
	for (k, v) in zip(locs, values):
		for k_sub in k:
			setDict(d, k_sub, v)
	setDict(d, ('simulation', 'result_location'), output_location + 'result-' + suffix + '.npz')

	with open(param_files[-1], 'w') as f:
		json.dump(d, f, indent=4)

def run_sim(script, param_file, basepath=''):
	return subprocess.call('/usr/local/python-2.7.6/bin/python '+basepath+script+' '+basepath+param_file, shell=True)

def run_sim_star(l):
	return run_sim(*l)

script = sys.argv[1]

try:
	mode = sys.argv[3]
except:
	mode = 'test'

if mode == 'local':

	import multiprocessing

	cpu_count = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=cpu_count)
	pool.map(run_sim_star, [(script, f) for f in param_files])

elif mode == 'spark':

	from pyspark import SparkConf, SparkContext

	with open(os.path.expanduser("~/spark-master")) as f:
		master = f.readline().replace('\n', '')

	conf = SparkConf().setAppName('sweep').setMaster(master)
	sc = SparkContext(conf=conf)

	basepath = os.path.abspath('.')+'/'
	n_sims = len(param_files)
	print sc.parallelize(param_files, numSlices=n_sims).map(lambda f: run_sim(script, f, basepath=basepath)).collect()

elif mode == 'test':
	pass

with open(output_location+'ordering.npz', 'w') as f:
	np.savez(f, params=np.array(locs, dtype='object'), values=np.array(vals))

for name in param_files:
	#subprocess.check_output('rm ' + name, shell=True)
	pass

