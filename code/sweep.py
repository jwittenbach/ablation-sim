import os, sys, json, subprocess
import numpy as np
from sweep_utils import makeDicts, setDict

# get parameters from config file
script, fname = sys.argv[1:3]
with open(fname, 'r') as f:
	params = json.loads(f.read())

# create a new dictionary for every parameter setting
inds, dicts, locations, values = makeDicts(params, verbose=False)

output_location = os.path.abspath(params["simulation"]["result_location"])
if output_location[-1] != '/':
	output_location += '/'

param_files = []
for (idx, d) in zip(inds, dicts):

	suffix = '-'.join([str(i) for i in idx])
	param_files.append('.params-' + suffix + ".json")
	
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
	np.savez(f, params=np.array(locations, dtype='object'), values=np.array(values))

if mode != 'test':
	for name in param_files:
		subprocess.check_output('rm ' + name, shell=True)