#! /usr/bin/env python

import brian2 as b2
import numpy as np

import sys, json
from copy import deepcopy
from scipy.stats import beta

from utils import Bunch, xc_score, spike_score, psp_to_current, ablateNeuron


# get parameters from config file
fname = sys.argv[1]
f = open(fname, 'r')
params = Bunch(json.loads(f.read()))

# set up simulation time
duration = params.simulation.duration * b2.second
dt = params.simulation.dt * b2.ms

# make neuron groups
model = '''
		dx/dt = (xinf - x + IsynE + IsynI + Iext(t, i))/tau: 1 (unless refractory)
    	dIsynE/dt = -IsynE/tauSynE: 1
     	dIsynI/dt = -IsynI/tauSynI: 1
     	xinf: 1
     	tau: second
     	tauSynE: second
     	tauSynI: second
     	'''

Nc = params.network.Nc
Nu = params.network.Ne - params.network.Nc
Ne = params.network.Ne
Ni = params.network.N - params.network.Ne
Ntotal = params.network.N
N = [Nc, Nu, Ni]

G = b2.NeuronGroup(Ntotal, model, threshold='x>theta', reset='x=reset', dt=dt, refractory=params.neurons.refractory*b2.ms)

Gc = G[:Nc]		#clustered (exc)
Gu = G[Nc:Ne]	#unclustered (exc)
Gi = G[Ne:]		#inhibitory
Glist = [Gc, Gu, Gi]

# if parameters to build network are supplied, build network
if type(params.network) is Bunch:
	
	#compute PSP sizes
	N_avg_connects = 1.0 * Ntotal * params.connectivity.pSyn
	N_to_spike = N_avg_connects * params.connectivity.frac_to_spike
	gap = params.neurons.theta - params.neurons.reset

	J = gap / N_to_spike
	
	Jee, Jei = J, J
	Jie, Jii = -params.connectivity.g*J, - params.connectivity.g*J

	#convert PSP sizes to synaptic strengths (kick sizes)
	tauE, tauI = params.neurons.tauE, params.neurons.tauI
	tauSynE, tauSynI = params.neurons.tauSynE, params.neurons.tauSynI

	Wee = psp_to_current(Jee, tauE/tauSynE)
	Wei = psp_to_current(Jei, tauI/tauSynE)
	Wie = psp_to_current(Jie, tauE/tauSynI)
	Wii = psp_to_current(Jii, tauI/tauSynI)

	#modify strengths
	Wee = params.connectivity.Wee_factor * Wee
	Wei = params.connectivity.Wei_factor * Wei
	Wie = params.connectivity.Wie_factor * Wie
	Wii = params.connectivity.Wii_factor * Wii

	Wcc = params.connectivity.Wcc_factor * Wee
	Wce = params.connectivity.Wce_factor * Wee
	Wec = params.connectivity.Wec_factor * Wee

	#create connections
	np.random.seed(params.connectivity.network_seed)
	connections = [[np.where(np.random.rand(N[i], N[j]) < params.connectivity.pSyn) for j in xrange(len(N))] for i in xrange(len(N))]
	weights = [[Wcc, Wce, Wei],
			   [Wec, Wee, Wei],
			   [Wie, Wie, Wii]]

# create synapses
pre_strings = ["IsynE+=w", "IsynI+=w"]
S = []
for i in xrange(len(N)):
	S.append([])
	for j in xrange(len(N)):
		S[i].append(b2.Synapses(Glist[i], Glist[j], model="w: 1", pre=pre_strings[int(i==2)], dt=dt, delay=params.synapses.synaptic_delay*b2.ms))
		S[i][j].connect(*connections[i][j])
		S[i][j].w = weights[i][j]

# set the seed that controls randomness in the simulation (stimulus jitter + initial conditions)
np.random.seed(params.simulation.seed)

# create ths stimulus
dur, step = duration/b2.second, dt/b2.ms
L = int(1000.0*dur/step)+1
Nstims = int(1000.0*dur/params.stimulus.interval)-1
stim_onsets = [params.stimulus.interval*i for i in np.arange(Nstims)+1]
stims = np.zeros((Ntotal, L))
a = params.stimulus.jitter.shape.a
b = params.stimulus.jitter.shape.b
peak0 = 1.0*(a-1)/(a+b-2)
stretch_factor = params.stimulus.jitter.peak/peak0
pulse = np.floor(params.stimulus.pulse_width/step)
strength1 = params.stimulus.strength
strength2 = params.stimulus.input_factor * strength1
for d in stim_onsets:
	for i in xrange(Ne):
		if i < Nc:
			strength = strength1
		else:
			strength = strength2
		jitter = stretch_factor * beta.rvs(a, b)
		loc = np.floor((d + jitter)/step)
		stims[i, loc:loc+pulse] = strength 

# initialize neurons
Iext = b2.TimedArray(stims.T, dt=dt)
theta = params.neurons.theta
reset = params.neurons.reset

G.x = np.random.rand(Ntotal)

G.xinf = params.neurons.xinf

Gc.tau = tauE*b2.ms
Gu.tau = tauE*b2.ms
Gi.tau = tauI*b2.ms

G.tauSynE = tauSynE*b2.ms
G.tauSynI = tauSynI*b2.ms

# create network
print "creating network"
Net = b2.Network(G, S)

print "creating monitors"
#monitor = b2.StateMonitor(G, 'x', record=True)
#monitorE = b2.StateMonitor(G, 'IsynE', record=True)
#monitorI = b2.StateMonitor(G, 'IsynI', record=True)
spikes = b2.SpikeMonitor(G)

print "adding monitors"
#Net.add(monitor)
#Net.add(monitorE)
#Net.add(monitorI)
Net.add(spikes)

# store initial configuration
print "storing initial state"
Net.store('initial')

# run simulation
"simulating network"
Net.run(duration, report='stdout')
spike_times_pre = np.array(spikes.t)
spike_ids_pre = np.array(spikes.i)

# compute scores
print "computing scores"
xc_scores_pre = xc_score(spikes, stims[0], duration, Ntotal, dt)
spike_scores_pre = spike_score(spikes, stim_onsets, duration, Ntotal, dt)

# write result
results = {}
results.update(spike_times_pre=spike_times_pre, spike_ids_pre=spike_ids_pre,
	xc_scores_pre=xc_scores_pre, spike_scores_pre=spike_scores_pre)
#	monitor_pre=np.asarray(monitor.x), monitorE_pre=np.asarray(monitorE.IsynE), monitorI_pre=np.asarray(monitorI.IsynI))

if params.ablate is not -1:
	# reset network
	print "restoring network to initial state"
	Net.restore('initial')

	# ablate high-scoring neurons
	print "ablating neurons"
	ablate = np.argsort(spike_scores_pre[:Ne].T)[::-1][:params.ablate]
	ablateNeuron(ablate, S, Nc)

	# rerun simulation
	print "simulating network -- take, the second"
	Net.run(duration, report='stdout')
	spike_times_post = np.array(spikes.t)
	spike_ids_post = np.array(spikes.i)

	# compute scores
	print "computing scores -- take, the second"
	xc_scores_post = xc_score(spikes, stims[0], duration, Ntotal, dt, ablate)
	spike_scores_post = spike_score(spikes, stim_onsets, duration, Ntotal, dt, ablate)

	results.update(spike_times_post=spike_times_post, spike_ids_post=spike_ids_post, 
		xc_scores_post=xc_scores_post, spike_scores_post=spike_scores_post, ablated=ablate)
		#monitor_post=np.asarray(monitor.x), monitorE_post=np.asarray(monitorE.IsynE), monitorI_post=np.asarray(monitorI.IsynI))

if not params.simulation.result_location == 0:
	print "writing results"
	np.savez(params.simulation.result_location, **results)
