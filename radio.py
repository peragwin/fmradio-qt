

from __future__ import division
import numpy as np
from numpy.fft import fft
from numpy.fft import ifft

#from numpy.fft import *
from scipy import signal
from rtlsdr import RtlSdr
import pyaudio

import matplotlib.pyplot as plt
#import time

from multiprocessing import Pool
#import threading



freq = 94.9e6

N_samples = 40960 # multiple of 256
#gpool = Pool(processes=4)
decim_r1 = 1e6/2e5# for wideband fm
decim_r2 = 2e5/44100 # for baseband recovery

    #audio_buffer = np.zeros(1807)
    #def __init__(self,freq):
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



hamming = 10*signal.hamming(N_samples*.10 )
lpf = np.append( np.zeros(N_samples*.45),hamming)
lpf = np.fft.fftshift(np.append(lpf,np.zeros(N_samples*.45)))

hamming = signal.hamming(N_samples*.05 )
lpf2 = np.append( np.zeros(N_samples*.075),hamming)
lpf2 = np.fft.fftshift(np.append(lpf2,np.zeros(N_samples*.075)))



def __del__():
    if sdr:
        print 'sdr_closed'
        sdr.close()

class Sampler:
    def __init__(self,sdr,N_samples):
        self.sdr = sdr
        self.ns = N_samples
        self.sample_buffer = np.zeros(self.ns)
    def getSamples(self):
        self.sample_buffer = self.sdr.read_samples(N_samples) #+ .5*self.sample_buffer
        return self.sample_buffer

#sampler = Sampler(sdr,N_samples)


def startSampling():
    sampler = Process(target=getSamples())

#       d1 = Value('f',np.zeros(samples.size*44100/1e6))
#       d2= Value('f',np.zeros(samples.size*44100/1e6))
    #d1 = Value('np.array')
    #d2 = Value('np.array')
 #   p1 = pool.apply_async(,s1)
 #   p2 = pool.apply_async(doub,s2)
    #print "not started"
  #  p1.start()
  #  p2.start()
    #print 'started'
 #   d1 = p1.get(timeout = 1)
 #   d2 = p2.get(timeout = 1)

def lowpass_filter(x,width):
        #wndw = np.sinc(np.r_[-15:16]/np.pi)/np.pi
        wndw = np.kaiser(width,6)
        wndw /= np.sum(wndw)
        new_array = signal.fftconvolve(x, wndw)

        return new_array[width/2:x.size+width/2]

def demodulate(args):
    samples = args#[0]
    #decim_r1 = args[1]
    #decim_r2 = args[2]
    # DEMODULATION CODE

    # LIMITER goes here
    
    # low pass & down sampling via fft
    lp_samples = signal.decimate(lowpass_filter(samples,32),int(decim_r1))
    #spectrum = fft(samples) * (lpf)

#         toplot = False
#         if(toplot):
#             fig = plt.figure()
#             plt.plot(np.abs(spectrum))
#             plt.show()

# Decimate in two rounds. One to 200k, another to 44.1k
    # DECIMATE HERE. Note that we're looking at 1MHz bandwidth.
    #n_s = spectrum.size
    #channel_spectrum = np.append(spectrum[0:n_s/decim_r1*.5],spectrum[n_s-n_s/decim_r1*.5:n_s])

    #radio_spectrum -= np.mean(radio_spectrum) #attempt to remove dc bias

#         toplot = False
#         if(toplot):
#             fig = plt.figure()
#             plt.plot(np.abs(channel_spectrum))
#             plt.show()


    #lp_samples = ifft(channel_spectrum)

    #lp_samples = lowpass_filter(lp_samples,4)[2:8194]

# polar discriminator

    A = lp_samples[1:lp_samples.size]
    B = lp_samples[0:lp_samples.size-1]

    dphase = ( A * np.conj(B) )

#dpm = np.mean(np.abs(dphase))
    # normalize
    #       dphase /= dpm
    dphase.resize(dphase.size+1)
    dphase[dphase.size-1] = dphase[dphase.size-2]

    rebuilt = signal.medfilt(np.angle(dphase)/np.pi,15) #  np.cos(dphase)

    #phase = np.sin(rebuilt)
    #phase = lowpass_filter(phase,8)

    #rebuilt= lowpass_filter(rebuilt,8)

    toplot = False
    if toplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(rebuilt)
        #ax.plot(phase)
        plt.show()

    
   # spectrum = fft(rebuilt) #* lpf2
   # n_s = spectrum.size
   # base_spectrum = np.append(spectrum[0:n_s/decim_r2*.5],spectrum[n_s-n_s/decim_r2*.5:n_s])
   # output = ifft(base_spectrum,n=base_spectrum.size)
    #check: should be 1807 or very close to it. it is!!
    output = signal.decimate(rebuilt,int(decim_r2))
    #output = lowpass_filter(np.real(output),16) #[12:8204]

#         toplot = False
#         if(toplot):
#             fig = plt.figure()
#             plt.plot(np.real(output))
#             plt.show()

    #print out.size
    return np.real(output)



def demodfork(samples):
   # print "demoding"
#     num_samples = N_samples
#     samples = np.zeros(num_samples)
#     i = 0
#     if not sample_buffer.empty():
#         while not sample_buffer.empty():
#             samples[i:i+N_samples] = sample_buffer.get()
    #samples = sample_buffer

    s1 = samples[0:samples.size:2]
    s2 = samples[1:samples.size:2]
    #s3 = samples[2:samples.size:4]
    #s4 = samples[3:samples.size:4]
    pool = Pool(processes=2)
    ans = pool.map(demodulate,[[s1,1e6/2e5,2e5/44100],
                               [s2,1e6/2e5,2e5/44100]])
                               #[s3,1e6/2e5,2e5/44100],
                               #[s4,1e6/2e5,2e5/44100]] )
    pool.close()
    #print "forked"
    reconst = np.zeros(2*ans[0].size)
    reconst[0:reconst.size:2] = ans[0]
    reconst[1:reconst.size:2] = ans[1] #+ans[2]+ans[3]
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


#         if not audio_buffer.empty():
#            # print "playing"
#             samples = audio_buffer.get()
#             while not audio_buffer.empty():
#                 samples = np.append(samples,audio_buffer.get())
        self.stream.write( samples.astype(np.float32).tostring() )




# def getAudio():
#     print "Listening to ",
#     print center_freq
#     while True:
#         audio_buffer.put(demodfork(getSamples()))
#         print "got it"
#         time.sleep(1)


def main():

    #radio = FMRadio(88.5e6)
    player = audioPlayer()
    sdr = SDR(freq)
    sampler = Sampler(sdr.sdr,N_samples)

#     pa = pyaudio.PyAudio()
#     stream = pa.open( format = pyaudio.paFloat32,
#                                     channels = 1,
#                                     rate = 44100,
#                                     output = True)
#     def play(samples):
#         stream.write( samples.astype(np.float32).tostring() )
    #sampler.daemon(True)
    #sampler.start()
    #sampler = Process(target=getSamples())
    #audio_buffer = np.zeros(18070/2)
   # i=0
    while True:
        player.play(demodulate(sampler.getSamples()))


main()


    #while True:
   #     if not audio_buffer.empty():
        # radio.getSamples()
#
#         sample_thread = threading.Thread()
#         p1_thread = theading.Thread(target=radio.demodulate()
    #        player.play(audio_buffer.get())
     #       print "played" #radio.demodfork(radio.getSamples()))


    #i = 0
    #while i < 500: # approx 20 seconds worth of crapy noise lol
    #    player.play(radio.demodulate(radio.getSamples()))
#    i+=1











#     def demodulate(self,samples):
#         # DEMODULATION CODE
#
#         # LIMITER goes here
#
#         # low pass & down sampling via fft
#
#         spectrum = fft(samples) * (lpf)
#
# #         toplot = False
# #         if(toplot):
# #             fig = plt.figure()
# #             plt.plot(np.abs(spectrum))
# #             plt.show()
#
# # Decimate in two rounds. One to 200k, another to 44.1k
#         # DECIMATE HERE. Note that we're looking at 1MHz bandwidth.
#         n_s = spectrum.size
#         channel_spectrum = np.append(spectrum[0:n_s/decim_r1*.5],spectrum[n_s-n_s/decim_r1*.5:n_s])
#
#         #radio_spectrum -= np.mean(radio_spectrum) #attempt to remove dc bias
#
# #         toplot = False
# #         if(toplot):
# #             fig = plt.figure()
# #             plt.plot(np.abs(channel_spectrum))
# #             plt.show()
#
#
#         lp_samples = ifft(channel_spectrum)
#
#         #lp_samples = lowpass_filter(lp_samples,4)[2:8194]
#
#     # polar discriminator
#
#         A = lp_samples[1:lp_samples.size]
#         B = lp_samples[0:lp_samples.size-1]
#
#         dphase = ( A * np.conj(B) )
#
# #dpm = np.mean(np.abs(dphase))
#         # normalize
#         #       dphase /= dpm
#         dphase.resize(dphase.size+1)
#         dphase[dphase.size-1] = dphase[dphase.size-2]
#
#         rebuilt = np.angle(dphase)/np.pi #  np.cos(dphase)
#
#         #phase = np.sin(rebuilt)
#         #phase = lowpass_filter(phase,8)
#
#         #rebuilt= lowpass_filter(rebuilt,8)
#
# #         toplot = False
# #         if toplot:
# #             fig = plt.figure()
# #             ax = fig.add_subplot(111)
# #             ax.plot(rebuilt)
# #             ax.plot(phase)
# #             plt.show()
#
#
#         spectrum = fft(rebuilt) #* lpf2
#         n_s = spectrum.size
#         base_spectrum = np.append(spectrum[0:n_s/decim_r2*.5],spectrum[n_s-n_s/decim_r2*.5:n_s])
#         output = ifft(base_spectrum,n=2*base_spectrum.size)
#         #check: should be 1807 or very close to it. it is!!
#
#         output = lowpass_filter(np.real(output),16) #[12:8204]
#
# #         toplot = False
# #         if(toplot):
# #             fig = plt.figure()
# #             plt.plot(np.real(output))
# #             plt.show()
#
#         #print out.size
#         return np.real(output)



#        marker19 = 1027 # calculated below
#        spsmax = np.max(radio_spectrum)
#        S = np.abs(np.fft.fftshift(radio_spectrum))
#        for i in range(0,radio_spectrum.size):
#            if S[i] > spsmax/2:
#                marker19 = i
#                print marker19
#                break;
#        scaleto = 19000/22050
#        shift = ((scaleto-1)*radio_spectrum.size + 2*marker19) / (2*scaleto+4)
#        print shift

#        shift = 0 #344.4 #103.4
#        adj_spectrum = np.fft.fftshift(radio_spectrum)[shift:radio_spectrum.size-shift]

#        marker19 = 1027 # calculated below
#        spsmax = np.max(radio_spectrum)
#        S = np.abs(np.fft.fftshift(radio_spectrum))
#        for i in range(0,radio_spectrum.size):
#            if S[i] > spsmax/2:
#                marker19 = i
#                print marker19
#                break;
#        scaleto = 19000/22050
#        print 2*marker19/adj_spectrum.size - (1-scaleto)




