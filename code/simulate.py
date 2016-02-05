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
     	theta: 1
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
	J = params.connectivity.J
	
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
	Wci = params.connectivity.Wci_factor * Wei
	Wic = params.connectivity.Wic_factor * Wie

	#print results
	print "J:\t" + str(J)
	print "Ji:\t" + str(Jii)
	print "Wee (init):\t" + str(psp_to_current(Jee, tauE/tauSynE))
	print "Wee (final):\t" + str(Wee)

	#create connections
	np.random.seed(params.connectivity.network_seed)
	pSyn = params.connectivity.pSyn
	pSyn[0][0] *= params.connectivity.pcc_factor
	connections = [[np.where(np.random.rand(N[i], N[j]) < pSyn[i][j]) for j in xrange(len(N))] for i in xrange(len(N))]
	weights = [[Wcc, Wce, Wci],
			   [Wec, Wee, Wei],
			   [Wic, Wie, Wii]]
	print 'weights:'
	for row in weights:
		print row
# create synapses
pre_strings = ["IsynE+=w", "IsynI+=w"]
S = []
r = 0.5
for i in xrange(len(N)):
	S.append([])
	for j in xrange(len(N)):
		#S[i].append(b2.Synapses(Glist[i], Glist[j], model="w: 1", pre=pre_strings[int(i==2)], dt=dt, delay=params.synapses.synaptic_delay*b2.ms))
		S[i].append(b2.Synapses(Glist[i], Glist[j], model="w: 1", pre=pre_strings[int(i==2)], dt=dt))
		S[i][j].connect(*connections[i][j])
		S[i][j].w = weights[i][j]
		Nsynapses = len(S[i][j].i)
		delays = params.synapses.synaptic_delay*(1+r*(2*np.random.rand(Nsynapses)-2))
		S[i][j].delay = delays*b2.ms

# create ths stimulus
print "setting up the stimulus"

a, b = params.stimulus.jitter.shape.a, params.stimulus.jitter.shape.b
peak0 = 1.0*(a-1)/(a+b-2)
stretch_factor = params.stimulus.jitter.peak/peak0
interval = params.stimulus.interval

print 'stretch factor:\t' + str(stretch_factor)

# compute stimulus time-course
step = dt/b2.ms
beta_duration = int(np.ceil(stretch_factor/step))
stim_duration = beta_duration
interval_duration = int(np.ceil(interval/step))
sim_duration = int(np.ceil(duration/dt))
stim = np.zeros(sim_duration)

single_stim = beta.pdf(1.0*np.arange(stim_duration)/(stim_duration-1), a, b)

i = 0
while True:
	start = i*interval_duration
	stop = start + stim_duration
	if stop > sim_duration:
		break
	stim[start:stop] += single_stim
	i += 1
stim = stim/(params.simulation.dt*np.sum(single_stim))

stim_onsets = np.arange(0, duration/b2.second, interval/1000.0)

if params.stimulus.stim_type == "pulsed":
	pulse_width = params.stimulus.pulse_width
	pulse_duration = int(np.ceil(pulse_width/step))

	np.random.seed(params.simulation.phase_seed)
	onsets_base = stretch_factor * beta.rvs(a, b, size=Ne)
	Ge.onsets = onsets_base + params.stimulus.jitter.variance * np.random.randn(Ne)
	Ge.offsets = Ge.onsets + pulse_width

	@b2.network_operation(dt=interval*b2.ms)
	def jitter_inputs(t):
		Ge.onsets = onsets_base + params.stimulus.jitter.variance * np.random.randn(Ne)
		Ge.offsets = Ge.onsets + pulse_width

	@b2.network_operation(dt=dt)
	def set_inputs(t):
		t_real = (t/b2.ms)
		if t_real < interval:
			pass
		else:
			t_interval = t_real % interval
			Ge.Iswitch = np.logical_and(Ge.onsets<t_interval, t_interval<Ge.offsets)

	Gc.Iext = params.stimulus.strength * params.stimulus.input_factor_c
	Gu.Iext = params.stimulus.strength * params.stimulus.input_factor_e
	Gi.Iext = 0

if params.stimulus.stim_type == "continuous":

	Gc.Iswitch = params.stimulus.strength * params.stimulus.input_factor_c
	Gu.Iswitch = params.stimulus.strength * params.stimulus.input_factor_e
	Gi.Iswitch = 0

	@b2.network_operation(dt=dt)
	def set_inputs(t):
		G.Iext = stim[int(10*(t/b2.ms))]

# create external drive
fExc = params.drive.fExc*np.ones(Ne)
fInh = params.drive.fInh*np.ones(Ni)
f = np.concatenate([fExc, fInh])*b2.Hz
P = b2.PoissonGroup(Ntotal, f, dt=dt)
Sdrive = b2.Synapses(P, G, model='w:1', pre='IsynE+=w', connect='i==j', dt=dt)
Sdrive.w = params.drive.str_factor*psp_to_current(Jee, tauE/tauSynE)
print "Wee:"
print Wee

# initialize neurons
np.random.seed(params.simulation.init_seed)
r = 0.5

#theta = params.neurons.theta
G.theta = params.neurons.theta*(1+r*(2*np.random.rand(Ntotal)-1))
#G.theta = params.neurons.theta
reset = params.neurons.reset

#G.x = 0.5*np.random.rand(Ntotal)
G.x = 0.5*params.neurons.theta*np.random.rand(Ntotal)
#G.x = 0

Ge.xinf = params.neurons.xinf_e
Gi.xinf = params.neurons.xinf_i
#Ge.xinf = params.neurons.xinf_e*(1+r*(2*np.random.rand(Ne)-1))
#Gi.xinf = params.neurons.xinf_i*(1+r*(2*np.random.rand(Ni)-1))

Ge.tau = tauE*b2.ms
Gi.tau = tauI*b2.ms

G.tauSynE = tauSynE*b2.ms
G.tauSynI = tauSynI*b2.ms

# create network
print "creating network"
Net = b2.Network(G, S, P, Sdrive)
if params.stimulus.stim_type == "pulsed":
	Net.add(jitter_inputs)
Net.add(set_inputs)

print "creating monitors"
if params.simulation.save_all:
	monitor = b2.StateMonitor(G, 'x', record=True)
	monitorE = b2.StateMonitor(G, 'IsynE', record=True)
	monitorI = b2.StateMonitor(G, 'IsynI', record=True)
	monitorIext = b2.StateMonitor(G, 'Iext', record=True)
	monitorIswitch = b2.StateMonitor(G, 'Iswitch', record=True)
spikes = b2.SpikeMonitor(G)

print "adding monitors"
if params.simulation.save_all:
	Net.add(monitor)
	Net.add(monitorE)
	Net.add(monitorI)
	Net.add(monitorIext)
	Net.add(monitorIswitch)
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
normalize = bool(params.simulation.corr_coef)
print "computing scores - xc"
xc_scores_pre = xc_score(spike_times_pre, spike_ids_pre, stim, duration/b2.second, Ntotal, dt/b2.second, normalize)
print "computing scores - spikes"
spike_scores_pre = spike_score(spike_times_pre, spike_ids_pre, stim_onsets, Ntotal, stretch_factor/1000.0)

# write result
results = {}
results.update(spike_times_pre=spike_times_pre, spike_ids_pre=spike_ids_pre, stimulus=stim,
	xc_scores_pre=xc_scores_pre, spike_scores_pre=spike_scores_pre)
if params.simulation.save_all:
	results.update(monitor_pre=np.asarray(monitor.x), monitorE_pre=np.asarray(monitorE.IsynE),
		monitorI_pre=np.asarray(monitorI.IsynI), monitorExt_pre=np.asarray(monitorIext.Iext)*np.asarray(monitorIswitch.Iswitch))

if params.ablate is not -1:
	# reset network
	print "restoring network to initial state"
	Net.restore('initial')

	# ablate high-scoring neurons
	print "ablating neurons"
	#ablate = np.argsort(spike_scores_pre[:Ne].T)[::-1][:params.ablate]
	#ablate = np.argsort(spike_scores_pre[:Nc].T)[::-1][:params.ablate]
	ablate = np.argsort(xc_scores_pre[:Ne].T)[::-1][:params.ablate]
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
	xc_scores_post = xc_score(spike_times_post, spike_ids_post, stim, duration/b2.second, Ntotal, dt/b2.second, normalize)
	spike_scores_post = spike_score(spike_times_post, spike_ids_post, stim_onsets, Ntotal, stretch_factor/1000.0)

	results.update(spike_times_post=spike_times_post, spike_ids_post=spike_ids_post, ablated=ablate,
		xc_scores_post=xc_scores_post, spike_scores_post=spike_scores_post)
	if params.simulation.save_all:
		results.update(monitor_post=np.asarray(monitor.x), monitorE_post=np.asarray(monitorE.IsynE),
			monitorI_post=np.asarray(monitorI.IsynI), monitorExt_post=np.asarray(monitorIext.Iext)*np.asarray(monitorIswitch.Iswitch))

if not params.simulation.result_location == 0:
	print "writing results"
	np.savez(params.simulation.result_location, **results)
