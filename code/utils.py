import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    return [i - np.sum(ablate<i) for i in inds]

def ratesFromSpikes(spikes, ids, i, duration, ker):   
    return fftconvolve(sp, ker, 'same')

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
    y_sampled = y[::sampleStep]
    score = maxCC(ccf(x_sampled, y_sampled), window)
    return np.nan_to_num(score)

def xc_score(spike_times, ids, stimulus, duration, N, dt):

    Nsteps = np.ceil(duration/dt)
    step = 1000*dt

    # make smoothing kernel
    sigma = 20
    L = 7*sigma
    t = np.arange(-L/2, L/2, step)
    ker = np.exp(-0.5*np.square(t/sigma))
    ker = ker/np.sum(ker)

    scores = np.zeros(N)
    for i in xrange(N):
        
        # find spike times
        inds = np.where(ids==i)[0]
        times = spike_times[inds]
        spike_train = np.zeros(Nsteps)
        inds = 1000.0*times/step
        spike_train[inds.astype('int')] = 1

        # get firing rate
        rate = fftconvolve(spike_train, ker, 'same')

        # get score
        scores[i] = score(stimulus, rate, 5, 100)

    return scores

def spike_score(spike_times, ids, onsets, N):
    window = 0.1
    offsets = onsets + window
    counts = np.zeros(N)

    for i in xrange(N):
        inds = np.where(ids==i)[0]
        times = spike_times[inds]
        c = 0
        for (start, stop) in zip(onsets, offsets):
            c += np.sum(np.logical_and(start<times, times<stop))
        counts[i] = c
    return 1.0*counts/len(onsets)



# --------------------------------------------------------

def plotSpikes(spike_times, spike_ids, duration=None, N=None, lines=[], ablated=[]):
    x = spike_times
    y = spike_ids
    
    inds = np.logical_not(np.in1d(y, ablated))
    x, y = x[inds], y[inds]
    
    for i in xrange(y.shape[0]):
        y[i] -= np.sum(ablated<y[i])
    
    plt.scatter(x, y, s=5, c='k')
    
    if duration is None:
        xmax = np.max(x)
    else:
        xmax = duration
    if N is None:
        ymax = max(y)
    else:
        ymax = N
    
    c = sns.color_palette()
    for i in xrange(len(lines)):
        plt.plot([0, xmax], [lines[i], lines[i]], '--', color=c[i], lw=4);
    plt.ylabel('neuron #')
    plt.xlabel('time (s)')

    plt.xlim([0, xmax])
    plt.ylim([0, ymax])
    return plt.gca()

def beforeAndAfterPlot(scores_pre, scores_post, inds, labels=None, jitter=None, limits=None, extra=False):
    
    n = len(inds)-1
    
    if labels is None:
        l = n*['']
    else:
        l = labels
    
    
    if jitter is not None:
        x = scores_valid + jitter*np.random.randn(len(scores_pre))
        y = scoresPost + jitter*np.random.randn(len(scores_post))
    else:
        x = scores_pre
        y = scores_post

    if limits is None:
        min_val = min((min(x), min(y)))-0.02
        max_val = max((max(x), max(y)))+0.02
    else:
        min_val, max_val = limits[0], limits[1]
    
    c = sns.color_palette('Set1', n+1)
    for i in xrange(n):
        i1, i2 = inds[i], inds[i+1]
        plt.scatter(x[i1:i2], y[i1:i2], c=c[i], label=l[i], lw=0, alpha=0.5)
        
    for i in xrange(n):
        i1, i2 = inds[i], inds[i+1]
        plt.scatter(np.mean(x[i1:i2]), np.mean(y[i1:i2]), c=c[i], lw=2, s=200, alpha=0.5)

    if extra is not None:
        plt.scatter(np.mean(x[inds[1]:]), np.mean(y[inds[1]:]), c=c[n], lw=2, s=200, alpha=0.5)
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.ylim([min_val, max_val])
    plt.xlim([min_val, max_val])
    
    plt.xlabel('score (pre)')
    plt.ylabel('score (post)')
    
    if labels is not None:
        plt.legend(loc='upper left')

