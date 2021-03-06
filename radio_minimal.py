

from __future__ import division
import numpy as np
#from numpy.fft import fft
#from numpy.fft import ifft

#from numpy.fft import *
from scipy import signal
from rtlsdr import RtlSdr
import pyaudio

#import matplotlib.pyplot as plt
#import time

from multiprocessing import Pool
#import threading



freq = 88.5e6

N_samples = 40960 # multiple of 256

decim_r1 = 1e6/2e5# for wideband fm
decim_r2 = 2e5/44100 # for baseband recovery


class SDR:
    def __init__(self,freq):
        self.sample_rate = 1e6

        self.center_freq = freq
        self.gain = 36

        self.sdr =  RtlSdr()
        self.sdr.direct_sampling = 1
        self.sdr.sample_rate = self.sample_rate
        self.sdr.center_freq = self.center_freq
        self.sdr.gain = self.gain
    def __del__(self):
        self.sdr.close()

class Sampler:
    def __init__(self,sdr,N_samples):
        self.sdr = sdr
        self.ns = N_samples
        self.sample_buffer = np.zeros(self.ns)
    def getSamples(self):
        self.sample_buffer = self.sdr.read_samples(N_samples) #+ .5*self.sample_buffer
        return self.sample_buffer

def lowpass_filter(x,width):
        #wndw = np.sinc(np.r_[-15:16]/np.pi)/np.pi
        wndw = np.kaiser(width,6)
        wndw /= np.sum(wndw)
        new_array = signal.fftconvolve(x, wndw)

        return new_array[width/2:x.size+width/2]

def demodulate(args):
    samples = args[0]
    decim_r1 = args[1]
    decim_r2 = args[2]
    # DEMODULATION CODE

    # LIMITER goes here

    # low pass & down sampling
    lp_samples = signal.decimate(lowpass_filter(samples,32),int(decim_r1))


# polar discriminator

    A = lp_samples[1:lp_samples.size]
    B = lp_samples[0:lp_samples.size-1]

    dphase = ( A * np.conj(B) )

    dphase.resize(dphase.size+1)
    dphase[dphase.size-1] = dphase[dphase.size-2]

    rebuilt = signal.medfilt(np.angle(dphase)/np.pi,15) #  np.cos(dphase)

    output = signal.decimate(rebuilt,int(decim_r2))

    return np.real(output)



def demodfork(samples):

    s1 = samples[0:samples.size/2]
    s2 = samples[samples.size/2:samples.size]
    #s3 = samples[2:samples.size:4]
    #s4 = samples[3:samples.size:4]
    pool = Pool(processes=2)
    ans = pool.map(demodulate,[[s1,1e6/2e5,2e5/44100],
                               [s2,1e6/2e5,2e5/44100]])

    pool.close()
    #print "forked"
    reconst = np.zeros(2*ans[0].size)
    reconst[0:reconst.size/2] = ans[0]
    reconst[reconst.size/2:reconst.size] = ans[1] #+ans[2]+ans[3]
    return reconst

# audio_buffer = Queue()



class audioPlayer:

    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open( format = pyaudio.paFloat32,
                                    channels = 1,
                                    rate = 44100,
                                    output = True)

    def play(self,samples):
        self.stream.write( samples.astype(np.float32).tostring() )

def main():

    #radio = FMRadio(88.5e6)
    player = audioPlayer()
    sdr = SDR(freq)
    sampler = Sampler(sdr.sdr,N_samples)

    while True:
        player.play(demodfork(sampler.getSamples()))

main()