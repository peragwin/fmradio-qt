import numpy as np
from scipy import signal

import matplotlib.pyplot as plt

from rtlsdr import RtlSdr

import pyaudio

import threading
import Queue


pa = pyaudio.PyAudio()
stream = pa.open( format = pyaudio.paFloat32,
         channels = 1,
         rate = 48000,
         output = True)

def play(samples):
   
    stream.write( samples.astype(np.float32).tostring() )
    

def gen_spec(x,m):
    itsreal = np.isreal(x[0])

    lx = x.size
    nt = (lx +m -1) // m
    xb = np.append(x,np.zeros(-lx+nt*m))
    xc = np.append(np.roll(x,int(m/2)),np.zeros(nt*m - lx))


    xr = np.reshape(xb, (m,nt), order='F') * np.outer(np.hanning(m),np.ones(nt))
    xs = np.reshape(xc, (m,nt), order='F') * np.outer(np.hanning(m),np.ones(nt))

    xm = np.zeros((m,2*nt),dtype='complex')
    xm[:,::2] = xr
    xm[:,1::2] = xs

    if itsreal:
        spec = np.fft.fftshift(np.fft.fft(xm,int(m/2),axis=0))
    else:
        spec = np.fft.fftshift(np.fft.fft(xm,m,axis=0))
    mx = np.max(spec)

    pwr = 64*(20* np.log(np.abs(spec)/mx + 1e-6)  + 60 )/60

    return np.real(pwr)

def show_image(im):
    asp = plt.figaspect(3.0/8)
    fig = plt.figure(figsize=asp, dpi=100)
    axes = fig.add_subplot(111)
    axes.imshow(im, aspect='auto', cmap='spectral')
    
    plt.show()






sdr = RtlSdr()
sdr.center_freq = 94.9e6
sdr.sample_rate = 2.4e5
sdr.gain = 22.9






#sdr.close()

#spectrogram = gen_spec(samples,4096)

#show_image(spectrogram)

sample_buffer = Queue.Queue(maxsize=10)

def sampler_callback(samples,context):
    sample_buffer.put(samples)

class MakeDaemon(threading.Thread):

    def __init__(self, function, args=None):
        threading.Thread.__init__(self)
        self.runnable = function
        self.args = args
        self.daemon = True

    def run(self):
        self.runnable()

def process_th():
    p = Processor()
    while(1):
        
        samples = sample_buffer.get()
       
        audio = p.process(samples)
        sample_buffer.task_done()

        play(audio)

class Processor:
    prevB = 0
    prevConv1 = np.zeros(256)
    prevConv2 = np.zeros(256)
    prevConv3 = np.zeros(128)
    def process(self,samples):

        #samples = sdr.read_samples(2.56e6)

        h = signal.firwin(256,80000,nyq=1.2e5)
        output = signal.fftconvolve(samples,h)
        output[:h.size/2] += self.prevConv1[h.size/2:]   #add the latter half of tail end of the previous convolution
        outputa = np.append(self.prevConv1[:h.size/2], output) # also delayed by half size of h so append the first half
        self.prevConv1 = output[output.size-h.size:]   # set the tail for next iteration
        lp_samples = outputa[:output.size-h.size]  # chop off the tail and decimate

        #lp_samples = output[::5]

        dmod = np.zeros(lp_samples.size)

        A = lp_samples[1:]
        B = lp_samples[:lp_samples.size-1]

        dmod[1:] = np.real(np.angle(A * np.conj(B))) / (np.pi)
        dmod[0] = np.real(np.angle(lp_samples[0] * np.conj(self.prevB))) / (np.pi)
        self.prevB = lp_samples[lp_samples.size-1]



        h = signal.firwin(256,1.6e4,nyq=1.2e5)
        output = signal.fftconvolve(dmod,h)
        output[:h.size/2] += self.prevConv2[h.size/2:]   #add the latter half of tail end of the previous convolution
        outputa = np.append(self.prevConv2[:h.size/2], output) # also delayed by half size of h so append the first half
        self.prevConv2 = output[output.size-h.size:]   # set the tail for next iteration
        audible = outputa[:output.size-h.size:5]  # chop off the tail and decimate
    

        #h = signal.firwin(128,1.6e4,nyq=24000)
        #output = signal.fftconvolve(audible,h)
        #output[:h.size/2] += prevConv3[h.size/2:]   #add the latter half of tail end of the previous convolution
        #outputa = np.append(prevConv3[:h.size/2], output) # also delayed by half size of h so append the first half
        #prevConvo3 = output[output.size-h.size:]   # set the tail for next iteration
        #audible = outputa[:output.size-h.size:5]  # chop off the tail and decimate
    
        #print audible.size
        #spec = gen_spec(audible,256)
        #show_image(spec)

        return np.real(audible)

def sampler():
    sdr.read_samples_async(sampler_callback,32768)


process_d = MakeDaemon(process_th)
process_d.start()

sampler()
#sampler_t = threading.Thread(target=sampler)
#sampler_t.start()

#now do other things

#sdr.close()