import numpy as np
from pyspark import SparkConf, SparkContext
import os

with open(os.path.expanduser("~/spark-master")) as f:
	master = f.readline().replace('\n', '')

conf = SparkConf().setAppName('sweep').setMaster(master)
sc = SparkContext(conf=conf)
sc.addPyFile('/tier2/freeman/Wittenbach/simulations/ablation-sim/code/utils.py')


def get_scores(filename):

	from utils import xc_score, spike_score, newInds

	Nstims = 70
	interval = 0.3
	inds_pre = [0, 200, 4000, 5000]

	data = np.load(filename)
	spike_times_pre, spike_ids_pre = data['spike_times_pre'], data['spike_ids_pre']
	spike_times_post, spike_ids_post = data['spike_times_post'], data['spike_ids_post']
	
	ablated = data['ablated']
	inds_post = newInds(inds_pre, ablated)

	scores_pre = np.zeros((len(inds_pre), Nstims))
	scores_post = np.zeros((len(inds_post), Nstims))
	onsets = np.array([0.0])

	for i in xrange(Nstims):
		print i
		
		all_scores = spike_score(spike_times_pre, spike_ids_pre, onsets, inds_pre[-1])
		all_scores = np.delete(all_scores, ablated)
		scores_pre[:len(inds_pre)-1, i] = [np.mean(all_scores[inds_pre[j]:inds_pre[j+1]]) for j in xrange(len(inds_pre)-1)]
		scores_pre[len(inds_pre)-1, i] = np.mean(all_scores[inds_pre[0]:inds_pre[2]])

		all_scores = spike_score(spike_times_post, spike_ids_post, onsets, inds_post[-1])
		all_scores = np.delete(all_scores, ablated)
		scores_post[:len(inds_post)-1, i] = [np.mean(all_scores[inds_post[j]:inds_post[j+1]]) for j in xrange(len(inds_post)-1)]	
		scores_post[len(inds_post)-1, i] = np.mean(all_scores[inds_post[0]:inds_post[2]])

		onsets = np.append(onsets, onsets[-1]+interval)

	return np.vstack((scores_pre, scores_post))

Nsims = 100
basepath = os.path.abspath('.')+'/'
files = [basepath+'result-'+str(i)+'.npz' for i in xrange(Nsims)]

scores = np.array(sc.parallelize(files, numSlices=Nsims).map(get_scores).collect())
np.save('scores.npy', scores)