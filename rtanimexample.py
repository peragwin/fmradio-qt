import numpy as np
from scipy import signal

import matplotlib.pyplot as plt


from rtlsdr import RtlSdr

import pyaudio

import sys 
import threading
import Queue

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s


class MainWindow(QMainWindow):

    def __init__(self):
        # set up py audio
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format = pyaudio.paFloat32,
                              channels = 1,
                              rate = 48000,
                              output = True)
        # set up rtlsdr
        self.sdr = RtlSdr()
        self.sdr.center_freq = 94.9e6
        self.sdr.sample_rate = 2.4e6
        self.sdr.gain = 22.9
        
        # create the Qt ui
        self.parent.__init__()
        self.setupUi()
        # set up the main canvas for plotting stuff
        self.initCanvas()

        # create a sampler object. 
        # Note that sdr.read_samples_async creates its own background thread
        self.sampler = Sampler()
        self.sampler.beginSampling()

        # create a processor daemon (background process)
        self.process_d = MakeDaemon(self.process_th)
        self.process_d.start()

        


    def play(self,samples):
   
       self.stream.write( samples.astype(np.float32).tostring() )
    

    def draw_image(self,im):
        asp = plt.figaspect(3.0/8)
        fig = plt.figure(figsize=asp, dpi=100)
        axes = fig.add_subplot(111)
        axes.imshow(im, aspect='auto', cmap='spectral')
    
        plt.show()


    def process_th(self):
        p = Processor()
        while(1):
        
            samples = self.sampler.sample_buffer.get()
       
            processed = p.process(samples)
            sample_buffer.task_done()

            spec = self.gen_spec(processed,256)
            self.draw_image(spec)

    def initCanvas(self):
       
        self.dpi = 100
        self.fig = Figure((4.31,2.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.ui.plotFrame)

        self.axes = self.fig.add_subplot(111, aspect=200/431)
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        #self.axes.invert_yaxis()


    def gen_spec(self,x,m):
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

    def setupUi(self):
        
        self.setObjectName(_fromUtf8("MainWindow"))
        self.resize(729, 451)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))

        self.plotFrame = QWidget(self.centralwidget)
        self.plotFrame.setGeometry(QtCore.QRect(40, 40, 689, 411))
        self.plotFrame.setObjectName(_fromUtf8("plotFrame"))


class Sampler:
    def __init__(self,sdr,num_samples):
        self.num_samples
        self.sdr = sdr
        self.sample_buffer = Queue.Queue(maxsize=10)

    def sampler_callback(self,samples,context):
        self.sample_buffer.put(samples)

    def beginSampling():
        self.sdr.read_samples_async(self.sampler_callback,self.num_samples)



class Processor:
    prevB = 0
    prevConv1 = np.zeros(256)
    prevConv2 = np.zeros(256)
    prevConv3 = np.zeros(128)

    def process(self,samples):

        
        ##samples = sdr.read_samples(2.56e6)

        #h = signal.firwin(256,80000,nyq=1.2e5)
        #output = signal.fftconvolve(samples,h)
        #output[:h.size/2] += self.prevConv1[h.size/2:]   #add the latter half of tail end of the previous convolution
        #outputa = np.append(self.prevConv1[:h.size/2], output) # also delayed by half size of h so append the first half
        #self.prevConv1 = output[output.size-h.size:]   # set the tail for next iteration
        #lp_samples = outputa[:output.size-h.size]  # chop off the tail and decimate

        ##lp_samples = output[::5]

        #dmod = np.zeros(lp_samples.size)

        #A = lp_samples[1:]
        #B = lp_samples[:lp_samples.size-1]

        #dmod[1:] = np.real(np.angle(A * np.conj(B))) / (np.pi)
        #dmod[0] = np.real(np.angle(lp_samples[0] * np.conj(self.prevB))) / (np.pi)
        #self.prevB = lp_samples[lp_samples.size-1]



        #h = signal.firwin(256,1.6e4,nyq=1.2e5)
        #output = signal.fftconvolve(dmod,h)
        #output[:h.size/2] += self.prevConv2[h.size/2:]   #add the latter half of tail end of the previous convolution
        #outputa = np.append(self.prevConv2[:h.size/2], output) # also delayed by half size of h so append the first half
        #self.prevConv2 = output[output.size-h.size:]   # set the tail for next iteration
        #audible = outputa[:output.size-h.size:5]  # chop off the tail and decimate
    

        return np.real(audible)



class MakeDaemon(threading.Thread):

    def __init__(self, function, args=None):
        threading.Thread.__init__(self)
        self.runnable = function
        self.args = args
        self.daemon = True

    def run(self):
        self.runnable()


def main():

   app = QtGui.QApplication(sys.argv)
   frame = MainWindow()
   frame.show()
   app.exec_()

main()