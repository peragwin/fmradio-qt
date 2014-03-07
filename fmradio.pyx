

from __future__ import division
import numpy as np
#from numpy import fft
#from numpy.fft import *
from scipy import signal
from rtlsdr import RtlSdr
import pyaudio

import matplotlib.pyplot as plt

from multiprocessing.pool import ThreadPool
from multiprocessing import Queue

class FMRadio:

     # multiple of 256


    def __init__(self,freq,N_samples):

        self.sample_rate = 1e6
        self.decim_r1 = 1e6/2e5 # for wideband fm
        self.decim_r2 = 2e5/44100 # for baseband recovery
        self.center_freq = freq+250e3
        self.gain = 36

        self.N_samples = N_samples

        self.sdr =  RtlSdr()
        self.sdr.direct_sampling = 1
        self.sdr.sample_rate = self.sample_rate
        self.sdr.center_freq = self.center_freq
        self.sdr.gain = self.gain

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open( format = pyaudio.paFloat32,
                                    channels = 1,
                                    rate = 44100,
                                    output = True)

        adj = 0
        hamming = 10*signal.hamming(self.N_samples*.10 + adj)
        lpf = np.append( np.zeros(self.N_samples*.45),hamming)
        self.lpf = np.roll(np.fft.fftshift(np.append(lpf,np.zeros(self.N_samples*.45))),int(-.25*self.N_samples))

    def __del__(self):
        print "sdr closed"
        self.sdr.close()
        print "pyaudio terminated"
        self.pa.terminate()

    def getSamples(self):
        #N_samples = self.N_samples # 1/24.4 seconds   ~46336 #approximately a blocksize amount's time
        return self.sdr.read_samples(self.N_samples)


#    def demodulate_threaded(self,samples):
#        async_demodulation = self.pool.apply_async(self.demodulate, samples, callback=self.play)


    def demodulate(self,samples):
        # DEMODULATION CODE
        #samples = #self.sample_buffer.get()
        # LIMITER goes here

        # low pass & down sampling via fft

        spectrum = np.fft.fft(samples)*self.lpf

        toplot = False
        if(toplot):
            fig = plt.figure()
            plt.plot(np.abs(spectrum))
            plt.show()

# Decimate in two rounds. One to 200k, another to 44.1k
        # DECIMATE HERE. Note that we're looking at 1MHz bandwidth.
        n_s = spectrum.size

        channel_spectrum = spectrum[int(n_s*.75-.5*n_s/self.decim_r1):int(.75*n_s+.5*n_s/self.decim_r1)] #np.append(spectrum[0:int(n_s/self.decim_r1*.5)],spectrum[n_s-int(n_s/self.decim_r1*.5):n_s])

        #radio_spectrum -= np.mean(radio_spectrum) #attempt to remove dc bias
        #print channel_spectrum.size
        toplot = False
        if(toplot):
            fig = plt.figure()
            plt.plot(np.abs(np.fft.ifftshift(channel_spectrum)))
            plt.show()


        lp_samples = np.fft.ifft(np.fft.ifftshift(channel_spectrum))

        #lp_samples = self.lowpass_filter(lp_samples,4)

    # polar discriminator

        A = lp_samples[1:lp_samples.size]
        B = lp_samples[0:lp_samples.size-1]

        dphase = ( A * np.conj(B) )

#dpm = np.mean(np.abs(dphase))
        # normalize
        #       dphase /= dpm
        dphase.resize(dphase.size+1)
        dphase[dphase.size-1] = dphase[dphase.size-2]

        rebuilt = signal.medfilt(np.angle(dphase)/np.pi,25) #  np.cos(dphase)

        #phase = np.sin(rebuilt)
        #phase = self.lowpass_filter(phase,8)

        #rebuilt= self.lowpass_filter(rebuilt,8)

        toplot = False
        if toplot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(rebuilt)

            plt.show()


        spectrum = np.fft.fft(rebuilt) #* self.lpf2

        n_z = spectrum.size
        #base_spectrum = np.append(spectrum[0:1024],spectrum[n_z-1024:n_z])
        base_spectrum = np.append(spectrum[0:int(n_z/self.decim_r2*.5)],spectrum[n_z-int(n_z/self.decim_r2*.5):n_s])
        output = np.fft.ifft(base_spectrum)
        #check: should be 1807 or very close to it. it is!!

        output = self.lowpass_filter(np.real(output),8) #[12:8204]

#         toplot = False
#         if(toplot):
#             fig = plt.figure()
#             plt.plot(np.real(output))
#             plt.show()

        #print 1e6/n_s / 44100 * output.size
        return np.real(output)


    def demodulate2(self,samples):
        # DEMODULATION CODE

        # LIMITER goes here

        # low pass & down sampling
        lp_samples = signal.decimate(self.lowpass_filter(samples,16),int(self.decim_r1))


    # polar discriminator

        A = lp_samples[1:lp_samples.size]
        B = lp_samples[0:lp_samples.size-1]

        dphase = ( A * np.conj(B) )

        dphase.resize(dphase.size+1)
        dphase[dphase.size-1] = dphase[dphase.size-2]

        rebuilt = signal.medfilt(np.angle(dphase)/np.pi,15) #  np.cos(dphase)

        output = signal.decimate(rebuilt,int(self.decim_r2))

        return np.real(output)

    def lowpass_filter(self,x,width):
        #wndw = np.sinc(np.r_[-15:16]/np.pi)/np.pi
        wndw = np.kaiser(width,6)
        wndw /= np.sum(wndw)
        new_array = signal.fftconvolve(x, wndw)

        return new_array[int(width/2):x.size+int(width/2)]



    def play(self,samples):
        self.stream.write( samples.astype(np.float32).tostring() )

    def start(self):
        while True:
            self.play(self.demodulate(self.getSamples()))


def main():
    freq = 90.7e6
    radio = FMRadio(freq,32768*2)
    print "Currently listening to: ",
    print freq/1e6,
    print "MHz"
    radio.start()

main()
