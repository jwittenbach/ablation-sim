#! /usr/bin/env python

import brian2 as b2
import numpy as np

import sys, json
from copy import deepcopy
from scipy.stats import beta

from utils import Bunch, xc_score, spike_score, psp_to_current, ablateNeuron


# get parameters from config file
print "loading parameters"
fname = sys.argv[1]
f = open(fname, 'r')
params = Bunch(json.loads(f.read()))

# set up simulation time
duration = params.simulation.duration * b2.second
dt = params.simulation.dt * b2.ms

# make neuron groups
print "building neuron model"
model = '''
		dx/dt = (xinf - x + IsynE + IsynI + Iswitch*Iext)/tau: 1 (unless refractory)
    	dIsynE/dt = -IsynE/tauSynE: 1
     	dIsynI/dt = -IsynI/tauSynI: 1
     	xinf: 1
     	tau: second
     	tauSynE: second
     	tauSynI: second
     	onsets: 1
     	offsets: 1
     	Iext: 1
     	Iswitch: 1
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
Ge = G[:Ne]		#excitatory
Gi = G[Ne:]		#inhibitory
Glist = [Gc, Gu, Gi]

# if parameters to build network are supplied, build network
if type(params.network) is Bunch:
	
	print "building network"
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

# create ths stimulus
print "setting up the stimulus"
a, b = params.stimulus.jitter.shape.a, params.stimulus.jitter.shape.b
peak0 = 1.0*(a-1)/(a+b-2)
stretch_factor = params.stimulus.jitter.peak/peak0
interval = params.stimulus.interval
pulse_width = params.stimulus.pulse_width

np.random.seed(params.simulation.phase_seed)
Ge.onsets = stretch_factor * beta.rvs(a, b, size=Ne)
Ge.offsets = Ge.onsets + pulse_width

@b2.network_operation(dt=dt)
def set_inputs(t):
	t_interval = (t/b2.ms) % interval
	Ge.Iswitch = np.logical_and(Ge.onsets<t_interval, t_interval<Ge.offsets)

Gc.Iext = params.stimulus.strength * params.stimulus.input_factor_c
Gu.Iext = params.stimulus.strength * params.stimulus.input_factor_e
Gi.Iext = 0

# compute average (across neurons) stimulus
step = dt/b2.ms
beta_duration = int(np.ceil(stretch_factor/step))
pulse_duration = int(np.ceil(pulse_width/step))
stim_duration = beta_duration + pulse_duration
interval_duration = int(np.ceil(interval/step))
pulse = np.ones(pulse_duration)
single_stim = np.zeros(stim_duration)

#for i in xrange(beta_duration):
#	single_stim[i:i+pulse_duration] += beta.pdf(i*step/stretch_factor, a, b)*pulse
for (start, stop) in zip(np.array(Ge.onsets), np.array(Ge.offsets)):
	single_stim[start:stop] += 1.0
single_stim /= np.sum(single_stim)

sim_duration = int(np.ceil(duration/dt))
stim = np.zeros(sim_duration)
stim0 = np.zeros(sim_duration)
extra = np.ceil(peak0*stretch_factor/step)

i = 0
while True:
	start = i*interval_duration
	stop = start + stim_duration
	if stop > sim_duration:
		break
	stim[start:stop] += single_stim
	stim0[start+extra:start+extra+pulse_duration] = pulse
	i += 1

stim_onsets = np.arange(0, duration/b2.second, interval/1000.0)

# initialize neurons
theta = params.neurons.theta
reset = params.neurons.reset

np.random.seed(params.simulation.init_seed)
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
Net.add(set_inputs)

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
print "computing scores - xc"
xc_scores_pre = xc_score(spike_times_pre, spike_ids_pre, stim, duration/b2.second, Ntotal, dt/b2.second)
print "computing scores - spikes"
spike_scores_pre = spike_score(spike_times_pre, spike_ids_pre, stim_onsets, Ntotal)

# write result
results = {}
results.update(spike_times_pre=spike_times_pre, spike_ids_pre=spike_ids_pre, stimulus=stim,
	xc_scores_pre=xc_scores_pre, spike_scores_pre=spike_scores_pre)
#	monitor_pre=np.asarray(monitor.x), monitorE_pre=np.asarray(monitorE.IsynE), monitorI_pre=np.asarray(monitorI.IsynI))

if params.ablate is not -1:
	# reset network
	print "restoring network to initial state"
	Net.restore('initial')

	# ablate high-scoring neurons
	print "ablating neurons"
	ablate = np.argsort(spike_scores_pre[:Ne].T)[::-1][:params.ablate]
	#ablate = np.arange(params.ablate)
	ablateNeuron(ablate, S, Nc)

	print str(sys.argv[1]) + ": " + str(ablate)

	# rerun simulation
	print "simulating network -- take, the second"
	Net.run(duration, report='stdout')
	spike_times_post = np.array(spikes.t)
	spike_ids_post = np.array(spikes.i)

	# compute scores
	print "computing scores -- take, the second"
	xc_scores_post = xc_score(spike_times_post, spike_ids_post, stim, duration/b2.second, Ntotal, dt/b2.second)
	spike_scores_post = spike_score(spike_times_post, spike_ids_post, stim_onsets, Ntotal)

	results.update(spike_times_post=spike_times_post, spike_ids_post=spike_ids_post, ablated=ablate,
		xc_scores_post=xc_scores_post, spike_scores_post=spike_scores_post)
		#monitor_post=np.asarray(monitor.x), monitorE_post=np.asarray(monitorE.IsynE), monitorI_post=np.asarray(monitorI.IsynI))

if not params.simulation.result_location == 0:
	print "writing results"
	np.savez(params.simulation.result_location, **results)
