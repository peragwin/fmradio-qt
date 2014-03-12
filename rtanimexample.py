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
from PyQt4.QtGui import QWidget
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import ticker #used to get rid of axes labels
from matplotlib import animation

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
        self.sdr = RtlSdrSampler()
        self.sdr.num_samples = 32768
        self.sdr.center_freq = 94.9e6
        self.sdr.sample_rate = 1.2e6
        self.sdr.gain = 22.9
        
        # create the Qt ui
        QMainWindow.__init__(self)
        self.setupUi()

        #self.spec = np.zeros((820,600))

       
        
        print 'finished setting up'
        # create a sampler object. 
        # Note that sdr.read_samples_async creates its own background thread
        self.sdr.beginSampling()
        

        # create a processor daemon (background process)
        self.process_d = MakeDaemon(self.process_th)
        self.process_d.start()
        print 'finished init'
        
        # set up the main canvas for plotting stuff
        # doing this last because p.spec needs to be set up before drawing it
        self.initCanvas()

        


    def play(self,samples):
   
       self.stream.write( samples.astype(np.float32).tostring() )
    

    def draw_image(self,idk):
        self.axes.clear()
        self.axes.imshow(self.p. spec, cmap='spectral')
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        self.axes.set_aspect('auto',adjustable='box',anchor='NW')
        self.canvas.draw()


    def process_th(self):
        self.p = Processor()
        while(1):
            try:
                samples = self.sdr.sample_buffer.get()
            except:
                break # used to end process when program terminates
            processed = self.p.process(samples)
          
            self.sdr.sample_buffer.task_done()
            self.play(processed)
           # self.spec = np.roll(self.spec,1,axis = 1)
            #self.spec[:,0] = processed[0]
            
            #self.draw_image(spec)
            

    def initCanvas(self):
        self.main_frame = QWidget()

        self.dpi = 100
        self.fig = Figure((7.29,4.20), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        self.setCentralWidget(self.main_frame)

        self.axes = self.fig.add_subplot(111) #, aspect=200/431)
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        #self.axes.invert_yaxis

        self.anim = animation.FuncAnimation(self.fig,self.draw_image,interval=100,blit=False)
        
        


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
        #xm=xr

        if itsreal:
            spec = np.fft.fft(xm,m,axis=0)[int(m/2):]
        else:
            spec = np.fft.fftshift(np.fft.fft(xm,m,axis=0))
        mx = np.max(spec)

        pwr = 64*(20* np.log(np.abs(spec)/mx + 1e-6)  + 60 )/60

        return np.real(pwr)

    def setupUi(self):
       
        self.setWindowTitle('Demo: Real Time with RtlSdr')
       
        self.setObjectName(_fromUtf8("MainWindow"))
        self.resize(729, 451)
        #self.centralwidget = QtGui.QWidget(self)
        #self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        #self.setCentralWidget(self.centralwidget)

        #self.plotFrame = QWidget(self.centralwidget)
        #self.plotFrame.setGeometry(QtCore.QRect(40, 40, 689, 411))
        #self.plotFrame.setObjectName(_fromUtf8("plotFrame"))
    def __del__(self):
        # Program will continue running in the background unless the RTL is told to stop sampling
        self.sdr.cancel_read_async()
        print "sdr closed"
        self.sdr.close()
        print "pyaudio terminated"
        self.pa.terminate()
        
       # sys.exit()


class RtlSdrSampler(RtlSdr):
    num_samples = 32768 # default
    def __init__(self):
        RtlSdr.__init__(self)
        # create a Queue for read_samples_async       
        self.sample_buffer = Queue.Queue(maxsize=10)

    def sampler_callback(self,samples,context):
        self.sample_buffer.put(samples)

    def beginSampling(self):
        self.sample_t = threading.Thread(target=self.read_samples_async,args=(self.sampler_callback,self.num_samples))
        self.sample_t.start()
        print 'started sampling async'



class Processor:
    prevB = 0
    prevConv1 = np.zeros(256)
    prevConv2 = np.zeros(256)
    prevConv3 = np.zeros(128)

    spec = np.zeros((820,600))

    def process(self,samples):

        h = signal.firwin(256,80000,nyq=1.2e6/2)
        output = signal.fftconvolve(samples,h)
        output[:h.size/2] += self.prevConv1[h.size/2:]   #add the latter half of tail end of the previous convolution
        outputa = np.append(self.prevConv1[:h.size/2], output) # also delayed by half size of h so append the first half
        self.prevConv1 = output[output.size-h.size:]   # set the tail for next iteration
        lp_samples = outputa[:output.size-h.size:5]  # chop off the tail and decimate

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
    
        spectrum = np.log(np.abs( np.fft.fftshift(np.fft.fft(samples))))[::40] #smaller spectrum to handle when drawing
        self.spec = np.roll(self.spec,1,axis = 1)
        self.spec[:,0] = spectrum

        return np.real(.5*audible)



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
   frame.__del__()

main()