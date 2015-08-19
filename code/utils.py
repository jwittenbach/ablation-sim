import numpy as np
import brian2 as b2
from scipy.signal import fftconvolve

class Bunch(object):
    '''
    A class that wraps a dictionary and provides attribute-style access
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)
        for (k, v) in self.__dict__.iteritems():
            if type(v) is dict:
                self.__dict__[k] = Bunch(v)
    def __repr__(self):
        return self.__dict__.__str__()

def psp_to_current(psp, tauRatio):
    ''' 
    Function to convert PSP sizes into current step sizes

    Assuming a kick-and-decay synaptic current, determines the size of the current
    "kick" given the desired size of the PSP. the computation also requires the ratio
    between the two relevant decay time constants: the (slower) time constant on the
    membrane potential, and the (faster) time constant on the synaptic current.
    '''
    a = tauRatio
    c = np.power(a, 1.0/(1-a))-np.power(a, 1.0/(1.0/a-1))
    return psp*(a-1)/c

def ablateNeuron(ablate, S, Nc):
    '''
    Function to ablate neurons by removing their synapses
    '''

    for idx in ablate:
        
        if idx > Nc:
            idx -= Nc
            group = 1
        else:
            group = 0

        for i in xrange(len(S)):

            # neuron to be removed is presynaptic
            weights = np.array(S[group][i].w)
            pre = np.array(S[group][i].i)

            inds = np.where(pre==idx)[0]
            weights[inds] = 0

            S[group][i].w = weights

            # neuron to be removes is postsynaptic
            weights = np.array(S[i][group].w)
            post = np.array(S[i][group].j)

            inds = np.where(post==idx)[0]
            weights[inds] = 0

            S[i][group].w = weights

def newInds(inds, ablate):
    ablateGrouped = [np.sum(np.logical_and(ablate<inds[i+1], ablate>=inds[i])) for i in xrange(len(inds)-1)]
    return np.array(inds) - np.cumsum(np.insert(ablateGrouped, 0, 0))

def getSpikeArray(spikes, duration, N, dt, ablated=[]):
    N -= len(ablated)
    spikeArray = np.zeros((N, duration/dt))
    for (i, t) in zip(np.array(spikes.i), 10*spikes.t/b2.ms):
        if i in ablated:
            continue
        i -= sum(np.asarray(ablated) < i)
        spikeArray[i, int(t)] = 1
    return spikeArray

def ratesFromSpikes(spikeArray, ker):   
    rates = np.array([fftconvolve(sp, ker, 'same') for sp in spikeArray])
    return rates

def spikeCounts(spikesArray):
    return np.sum(spikesArray, axis=1)

def ccf(x, y, axis=None):
    assert x.ndim == y.ndim, "Inconsistent shape !"
#    assert(x.shape == y.shape, "Inconsistent shape !")
    if axis is None:
        if x.ndim > 1:
            x = x.ravel()
            y = y.ravel()
        npad = x.size + y.size
        xanom = (x - x.mean(axis=None))
        yanom = (y - y.mean(axis=None))
        Fx = np.fft.fft(xanom, npad, )
        Fy = np.fft.fft(yanom, npad, )
        iFxy = np.fft.ifft(Fx.conj() * Fy).real
        varxy = np.sqrt(np.inner(xanom, xanom) * np.inner(yanom, yanom))
    else:
        npad = x.shape[axis] + y.shape[axis]
        if axis == 1:
            if x.shape[0] != y.shape[0]:
                raise ValueError("Arrays should have the same length!")
            xanom = (x - x.mean(axis=1)[:, None])
            yanom = (y - y.mean(axis=1)[:, None])
            varxy = np.sqrt((xanom * xanom).sum(1) *
                            (yanom * yanom).sum(1))[:, None]
        else:
            if x.shape[1] != y.shape[1]:
                raise ValueError("Arrays should have the same width!")
            xanom = (x - x.mean(axis=0))
            yanom = (y - y.mean(axis=0))
            varxy = np.sqrt((xanom * xanom).sum(0) * (yanom * yanom).sum(0))
        Fx = np.fft.fft(xanom, npad, axis=axis)
        Fy = np.fft.fft(yanom, npad, axis=axis)
        iFxy = np.fft.ifft(Fx.conj() * Fy, n=npad, axis=axis).real
    # We just turn the lags into correct positions:
    iFxy = np.concatenate((iFxy[len(iFxy) / 2:len(iFxy)],
                           iFxy[0:len(iFxy) / 2]))
    if varxy < 0.001:
        return np.zeros(len(iFxy))
    else:
        return iFxy / varxy
        
def maxCC(signal, window):
    middle = len(signal)/2
    return max(signal[middle-window:middle+window])

def score(x, y, sampleStep, window):
    x, y = np.squeeze(x), np.squeeze(y)
    x_sampled = x[::sampleStep]
    y_sampled = y[:,::sampleStep]
    scores = np.array([maxCC(ccf(x_sampled, y_row), window) for y_row in y_sampled])
    return np.nan_to_num(scores)

def xc_score(spikes, stimulus, duration, N, dt, ablated=[]):
    # make spike array
    spikeArray = getSpikeArray(spikes, duration, N, dt, ablated)

    # make smoothing kernel
    sigma = 20
    L = 7*sigma
    t = np.arange(-L/2, L/2, dt/b2.ms)
    ker = np.exp(-0.5*np.square(t/sigma))
    ker = ker/np.sum(ker)

    # compute firing rates
    rates = ratesFromSpikes(spikeArray, ker)
    
    # compute scores
    return score(stimulus, rates, 5, 100)

def spike_score(spikes, onsets, duration, N, dt, ablated=[]):
    spikeArray = getSpikeArray(spikes, duration, N, dt, ablated)
    dt = dt/b2.ms
    onsets = np.floor(np.array(onsets)/dt)
    window = np.floor(100.0/dt)
    counts = np.array([np.sum(spikeArray[:, start:start+window], axis=1) for start in onsets])
    return np.mean(counts, axis=0)
