

from __future__ import division

# import math libraries
import numpy as np
from scipy import signal
from scipy.signal import signaltools as sigtool
from rtlsdr import RtlSdr
import pywt

import pyaudio

# Matplotlib files
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib import ticker

# System and control imports
import sys
import threading
import Queue
import time

from PyQt4 import QtCore, QtGui
from PyQt4.Qwt5 import QwtPlotCurve,QwtPlot

from radioui import Ui_MainWindow

# Helper functions generated by Qt Designer
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)



class FMRadio(QtGui.QMainWindow,Ui_MainWindow):
    sample_buffer = Queue.Queue(maxsize=10)
    base_spectrum = np.ones(5780)

    plotOverall = True
    plotChannel = False
    plotPlaying = False
    plotWaveform = False
    useStereo = False
    stereoWidth = 10
    useMedianFilt = True
    useLPFilt = True
    demodFiltSize = 11
    useAudioFilter = True
    audioFilterSize = 16
    toDraw = True
    demodMain = True
    demodSub1 = False
    demodSub2 = False
    toDrawWaterfalls = True
    toDrawPlots = False

    prevCutoff = 0

    prevConvo1 = np.zeros(128)

    toPlot = (np.cumsum(np.ones(5780)),np.cumsum(np.ones(5780)))

    def __init__(self,freq,N_samples):

        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.createQtConnections()
        ftxt = "%.1f MHz" % (freq/1e6)
        self.ui.curFreq.setText(ftxt)

        self.sample_rate = 2.4e5 ###1e6
        #self.decim_r1 = 1e6/2e5 # for wideband fm
        self.decim_r2 = 2.4e5/48000 # for baseband recovery
        self.center_freq = freq #+250e3
        self.gain = 22.9

        self.N_samples = N_samples
        self.is_sampling = False

        self.spectrogram = np.zeros((328,200))
        self.chspectrogram = np.zeros((328,200))
        self.plspectrogram = np.zeros((164,200))

        self.sdr =  RtlSdr()
        #self.sdr.direct_sampling = 1
        self.sdr.sample_rate = self.sample_rate
        self.sdr.center_freq = self.center_freq
        self.sdr.gain = self.gain

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open( format = pyaudio.paFloat32,
                                    channels = 2,
                                    rate = 48000,
                                    output = True)

        self.PLL = PhaseLockedLoop(self.N_samples,19000,self.sample_rate)

        self.noisefilt = np.ones(6554)
        b,a = signal.butter(1, 2122/48000*2*np.pi, btype='low')
        self.demph_zf = signal.lfilter_zi(b, a)

        adj = 0
        hamming = np.kaiser(self.N_samples/4 + adj,1)
        lpf = np.append( np.zeros(self.N_samples*3/8),hamming)
        self.lpf = np.fft.fftshift(np.append(lpf,np.zeros(self.N_samples*3/8))) #,int(-.25*self.N_samples))

        hamming = 10*signal.hamming(self.N_samples/16)
        lpf = np.append(np.zeros(self.N_samples*15/32),hamming)
        self.lpf_s1 = (np.append(lpf,np.zeros(int(self.N_samples*15/32))))
        #self.lpf_s1 = np.roll(temp,int(.5*self.N_samples*67/120))
        #self.lpf_s1 += np.roll(temp,int(-.5*self.N_samples*67/120))
        self.lpf_s1 = np.fft.fftshift(self.lpf_s1)
        #self.lpf_s1 += np.fft.fftshift(self.lpf_s1)

#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.plot(range(self.lpf_s1.size),self.lpf_s1)
#         fig.show()

        hamming = 10*signal.hamming(self.N_samples/32)
        lpf = np.append(np.zeros(self.N_samples*31/64),hamming)
        self.lpf_s2 = (np.append(lpf,np.zeros(int(self.N_samples*31/64))))
        #self.lpf_s2 = np.roll(temp,int(.5*self.N_samples*92/120))
        #self.lpf_s2 += np.roll(temp,int(-.5*self.N_samples*92/120))
        self.lpf_s2 = np.fft.fftshift(self.lpf_s2)

    def createQtConnections(self):

        QtCore.QObject.connect(self.ui.freqSelect, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), self.setFreq)
        QtCore.QObject.connect(self.ui.checkBox, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.setUseStereo)
        QtCore.QObject.connect(self.ui.mainchannel, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setDemodMain)
        QtCore.QObject.connect(self.ui.subband1, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setDemodSub1)
        QtCore.QObject.connect(self.ui.subband2, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setDemodSub2)
        QtCore.QObject.connect(self.ui.stereoWidthSlider, QtCore.SIGNAL(_fromUtf8("sliderMoved(int)")), self.setStereoWidth)
        QtCore.QObject.connect(self.ui.spectrum_overall, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setSpectrumOverall)
        QtCore.QObject.connect(self.ui.spectrum_channel, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setSpectrumChannel)
        QtCore.QObject.connect(self.ui.spectrum_playing, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setSpectrumPlaying)
        QtCore.QObject.connect(self.ui.spectrum_waveform, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setSpectrumWaveform)
        QtCore.QObject.connect(self.ui.demodFiltMedian, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setDemodFiltMedian)
        QtCore.QObject.connect(self.ui.demodFiltLP, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setDemodFiltLP)
        QtCore.QObject.connect(self.ui.demodFilterSize, QtCore.SIGNAL(_fromUtf8("sliderMoved(int)")), self.setDemodFiltSize)
        QtCore.QObject.connect(self.ui.audioFilterActive, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setAudioFiltUse)
        QtCore.QObject.connect(self.ui.audioFilterSizeSlider, QtCore.SIGNAL(_fromUtf8("sliderMoved(int)")), self.setAudioFiltSize)
        QtCore.QObject.connect(self.ui.exitButton, QtCore.SIGNAL(_fromUtf8("clicked()")), self.terminate)
        QtCore.QObject.connect(self.ui.drawPlot, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setDrawSpec)
        QtCore.QObject.connect(self.ui.waterfallButton, QtCore.SIGNAL(_fromUtf8('toggled(bool)')), self.setDrawWaterfalls)
      #  QtCore.QObject.connect(self.ui.plotButton, QtCore.SIGNAL(_fromUtf8("clicked(bool)")), self.setDrawPlot)

        self.bindPlot()


    def bindPlot(self):
        self.dpi = 100
        self.fig = Figure((4.31,2.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.ui.plotFrame)

        self.initplot()


    def initplot(self):
        self.axes = self.fig.add_subplot(111, aspect=200/431)
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        #self.axes.invert_yaxis()


    def replot(self,toPlot):
        self.axes.clear()
        self.axes.plot(toPlot[0],toPlot[1])
        self.axes.set_aspect('auto',anchor='C')
        self.canvas.draw()

    def setDrawSpec(self,s):
        self.toDraw = s

    def drawSpectrum(self):
        self.axes.clear()
        self.axes.imshow(self.spectrogram, cmap='spectral')
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        self.axes.set_aspect('auto',adjustable='box',anchor='NW')
        self.canvas.draw()

    def drawChspectrum(self):
        self.axes.clear()
        self.axes.imshow(self.chspectrogram, cmap='spectral')
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        self.axes.set_aspect('auto',adjustable='box',anchor='NW')
        self.canvas.draw()

    def drawPlspectrum(self):
        self.axes.clear()
        self.axes.imshow(self.plspectrogram, cmap='spectral')
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        self.axes.set_aspect('auto',adjustable='box',anchor='NW')
        self.canvas.draw()

    def setDrawPlots(self,s):
        self.toDrawPlots = s
        self.toDrawWaterfalls = not s

    def setDrawWaterfalls(self,s):
        self.toDrawWaterfalls = s
        self.toDrawPlots = not s

    def setFreq(self,freq):
        if freq % 2 == 0:
            freq += 1
        freq /= 10.0
        text = "%.1f MHz" % freq
        self.ui.curFreq.setText(text)
        self.center_freq = freq*1e6 #+ 250e3
        setf_t = threading.Thread(target=self.setF_th, args=[self.center_freq,])
        setf_t.start()
        setf_t.join()



    # This function is what is used to adjust the tuner on the RTL
    # Currently, it causes the program to crash if used after an unspecified period of inactivity
    #     commented lines are attempts that didn't work
    def setF_th(self,f):
        while(self.is_sampling == True):
            pass
        #self.sdr.cancel_read_async()
        time.sleep(.1)
        self.sdr.center_freq = f
        #self.getSamplesAsync()

    def setUseStereo(self,u):
        self.useStereo = u
    def setStereoWidth(self,w):
        self.stereoWidth = np.sqrt(10*w)

    def setDemodMain(self,s):
        self.demodMain = s
        self.demodSub1 = not s
        self.demodSub2 = not s
        #self.useStereo = True
    def setDemodSub1(self,s):
        self.demodMain = not s
        self.demodSub1 = s
        self.demodSub2 = not s
        #self.useStereo = False
    def setDemodSub2(self,s):
        self.demodMain = not s
        self.demodSub1 = not s
        self.demodSub2 = s
        #self.useStereo = False

    def setSpectrumOverall(self,s):
        #self.initplot()
        self.plotOverall = s
        self.plotChannel = not s
        self.plotPlaying = not s
        self.plotWaveform = not s
    def setSpectrumChannel(self,s):
        #self.initplot()
        self.plotChannel = s
        self.plotOverall = not s
        self.plotPlaying = not s
        self.plotWaveform = not s
    def setSpectrumPlaying(self,s):
        #self.initplot()
        self.plotPlaying = s
        self.plotChannel = not s
        self.plotOverall= not s
        self.plotWaveform = not s
    def setSpectrumWaveform(self,s):
        self.plotWaveform = s
        self.plotPlaying = not s
        self.plotChannel = not s
        self.plotOverall= not s

    def setDemodFiltMedian(self,s):
        self.useMedianFilt = s
        self.useLPFilt = not s
    def setDemodFiltLP(self,s):
        self.useLPFilt = s
        self.useMedianFilt = not s
    def setDemodFiltSize(self,s):
        #if(s % 2 == 0):
        #    s+=1
        self.demodFiltSize = s

    def setAudioFiltUse(self,s):
        self.useAudioFilter = s
    def setAudioFiltSize(self,s):
        self.audioFilterSize = s

    def terminate(self):
        self.__del__()

    # Destructor - also used to exit the program when user clicks "Quit"
    def __del__(self):
        # Program will continue running in the background unless the RTL is told to stop sampling
        self.sdr.cancel_read_async()
        print "sdr closed"
        self.sdr.close()
        print "pyaudio terminated"
        self.pa.terminate()
        sys.exit()

    # Not currently used
    def getSamples(self):
        return self.sdr.read_samples(self.N_samples);

    def getSamplesAsync(self):
        #Asynchronous call. Initiates a continuous loop with the callback fn
        self.is_sampling = True
        samples = self.sdr.read_samples_async(self.sampleCallback,self.N_samples,context=self)
    def sampleCallback(self,samples,sself):

        self.is_sampling = False
        self.sample_buffer.put(samples)
        #print 'put some samples in the jar'

        # recursive loop
        #sself.getSamplesAsync()

    def demodulate_th(self):
        # Initiates a loop to process all the incoming blocks of samples from the Queue
        # This should be run in its own thread, or the program will become unresponsive
        while(1):

            try:
                samples = self.sample_buffer.get()
                #samples2 = self.sample_buffer.get()
            except:
                print "wtf idk no samples?"  # even though this can't happen... (although I'm not sure why not)
                print 'gonna try to finish off the to-do list'
                self.sample_buffer.join()
                break

            out1 = self.demodulate(samples)
            self.sample_buffer.task_done()
            #out2 = self.demodulate(samples2)
            #self.sample_buffer.task_done()

            audio_out = out1 #np.append(out1,out2)
            self.play(audio_out)


    def demodulate(self,samples):
        # DEMODULATION CODE - And the core function
        # samples must be passed in by the caller

        self.count += 1

        spectral_window = signal.blackmanharris

        spectrum = np.fft.fftshift(np.fft.fft(samples*spectral_window(samples.size)))

        self.spectrogram = np.roll(self.spectrogram, 1,axis=1)
        self.spectrogram[:,0] = np.log(np.abs(spectrum[::100]))

        if(self.toDraw and self.plotOverall and self.count % 10 == 9):
#             self.toPlot = (np.linspace(-5e5,5e5,spectrum.size),np.abs(spectrum))
#             self.replot()
            self.drawSpectrum()



        h = signal.firwin(128, 100000,nyq=1.2e5)
        lp_samples = signal.fftconvolve(samples,h,mode='same')


    # polar discriminator

        dphase = np.zeros(lp_samples.size, dtype='complex')

        A = lp_samples[1:lp_samples.size]
        B = lp_samples[0:lp_samples.size-1]

        dphase[1:] = ( A * np.conj(B) )

        dphase[0] = lp_samples[0] * np.conj(self.prevCutoff) #dphase[dphase.size-2]
        self.prevCutoff = lp_samples[lp_samples.size-1]

     # limiting
        mag = np.abs(dphase)
        dphase /= mag

#         if self.useMedianFilt:
#             rebuilt = signal.medfilt(np.angle(dphase)/np.pi,self.demodFiltSize) #  np.cos(dphase)
#         else:
#             rebuilt = self.lowpass(np.angle(dphase),self.demodFiltSize)
        rebuilt = np.angle(dphase) / np.pi
#         toplot = False
#         if toplot:
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#             ax.plot(rebuilt)

        power = np.abs(self.mad(lp_samples))
        self.ui.signalMeter.setValue(20*(np.log10(power)))

        demodMain = False
        demodSub1 = False
        demodSub2 = False
        if self.demodMain:
            demodMain = True

           # lp_samples = samples
        elif self.demodSub1:
            demodSub1 = True

            #lp_samples = samples * np.exp(-1j*2*np.pi*67650/2.4e5*np.r_[0:samples.size])
        elif self.demodSub2:
            demodSub2 = True
            #lp_samples = samples * np.exp(-1j*2*np.pi*92000/2.4e5*np.r_[0:samples.size])


        spectrum = np.fft.fft(rebuilt) #*spectral_window(rebuilt.size))


         #toplot = self.plotChannel
        self.chspectrogram = np.roll(self.chspectrogram, 1,axis=1)
        self.chspectrogram[:,0] = np.log(np.abs(spectrum[spectrum.size/2:spectrum.size:50]))
        if(self.toDraw and self.plotChannel and self.count % 10 == 9):
            self.drawChspectrum()
             #plotspectrum = np.abs(channel_spectrum[::100])
             #self.toPlot = (np.linspace(-np.pi,np.pi,plotspectrum.size),plotspectrum)
             #self.replot()


        isStereo = False
        #demodSub1 = False
        #demodSub2 = False
        n_z = rebuilt.size
        if demodMain:

            h = signal.firwin(128,16000,nyq=1.2e5)
            output = signal.fftconvolve(rebuilt,h)
            outputa = output    # could be done in place but I'm not concerned with memory

            outputa[:h.size/2] += self.prevConvo1[h.size/2:]   #add the latter half of tail end of the previous convolution
            outputa = np.append(self.prevConvo1[:h.size/2], outputa) # also delayed by half size of h so append the first half

            self.prevConvo1 = output[output.size-h.size:]   # set the tail for next iteration

            output = outputa[:output.size-h.size:self.decim_r2]  # chop off the tail and decimate


            #stereo_spectrum = spectrum
            if self.useStereo:
                isStereo = True
            #    pilotfilt = np.zeros(spectrum.size)
                ss = spectrum.size
            #    pilotfilt[18000/1.2e5*ss:ss*20000/1.2e5] = 1
            #    pilotfilt[spectrum.size-ss*20000/1.2e5:spectrum.size-ss*18000/1.2e5] = 1
            #    pilot = np.fft.ifft(spectrum * pilotfilt)
            #    pimod = rebuilt*np.square(pilot)
              #  h = signal.firwin(128,16000,nyq=1.2e5)
              #  lpmod = signal.fftconvolve(pimod,h,mode='same')
             #   diff = lpmod[::self.decim_r2]

                pilot = rebuilt * np.cos(2*np.pi*19/240*(np.r_[0:rebuilt.size]))
                h = signal.firwin(256,[18000,20000],pass_zero=False,nyq=1.2e5)
                pilot_actual = signal.fftconvolve(pilot,h,mode='same')
                self.PLL.adjust(pilot_actual)
               
                moddif = rebuilt * self.PLL.pllx2() #np.cos(2*np.pi*38/240*(np.r_[0:ss] - phase_shift))
                h = signal.firwin(128,16000,nyq=1.2e5)
                moddif = signal.fftconvolve(moddif,h,mode='same')
 
                h = signal.firwin(64,16000,nyq=48000/2)
                diff = signal.fftconvolve(moddif[::self.decim_r2],h,mode='same')

               
            #self.base_spectrum = np.append(spectrum[0:int(n_z/self.decim_r2*.5)],spectrum[n_z-int(n_z/self.decim_r2*.5):n_z])
            #output = np.fft.ifft(self.base_spectrum)



#             h = signal.firwin(128,16000,nyq=2.4e4)
#             output = signal.fftconvolve(output,h,mode='same')
        elif demodSub1:
#2122/samplerate*2 *pi butterworth

            demod = rebuilt * np.exp(-2j*np.pi*67650/2.4e5*np.r_[0:rebuilt.size])
         #   spectrum = np.fft.fft(demod)*self.lpf_s1
            h = signal.firwin(128,7500,nyq=2.4e5/2)
            lp_demod = signal.fftconvolve(demod,h,mode='same')
            decim = lp_demod[::self.decim_r2]
        #    base_spectrum = np.append(spectrum[0:int(.5*n_z*7500/2.4e5)],spectrum[n_z-int(.5*n_z*7500/2.4e5):n_z])
        #    decim = np.fft.ifft(base_spectrum)

            dphase = np.zeros(decim.size, dtype='complex')
#
            A = decim[1:decim.size]
            B = decim[0:decim.size-1]
#
            dphase[1:] = np.angle( A * np.conj(B) )
            h = signal.firwin(128,7500,nyq=24000)
            output = signal.fftconvolve(dphase,h,mode='same')

            #retoutput)

        elif demodSub2:
            demod = rebuilt * np.exp(-2j*np.pi*92000/2.4e5*np.r_[0:rebuilt.size])
          #  spectrum = np.fft.fft(demod)*self.lpf_s2
            h = signal.firwin(128,7500,nyq=2.4e5/2)
            lp_demod = signal.fftconvolve(demod,h,mode='same')
            decim = lp_demod[::self.decim_r2]
          #  base_spectrum = np.append(spectrum[0:int(.5*n_z*7500/2.4e5)],spectrum[n_z-int(.5*n_z*7500/2.4e5):n_z])
          #  decim = np.fft.ifft(base_spectrum)

            dphase = np.zeros(decim.size, dtype='complex')
#
            A = decim[1:decim.size]
            B = decim[0:decim.size-1]
#
            dphase[1:] = np.angle( A * np.conj(B) )
            h = signal.firwin(128,7500,nyq=24000)
            output = signal.fftconvolve(dphase,h,mode='same')

            #return np.real(output)

        #dc filter, lol
        output -= np.mean(output)

        if self.count < 4:
            #print "error" # for some reason, output is NaN for the first 2 loops
            return np.zeros(6554)

        


       
########## Denoising Filter Attempts ###############
#        pwr = np.log(1e-60 + np.abs(spectrum))
#        if not np.abs(pwr[0]) > 0:
#            pwr = np.ones(6554)
#        self.noisefilt *= .5
#        self.noisefilt = np.add(self.noisefilt,pwr) #+ self.noisefilt * .001*np.ones(6554) # (self.demodFiltSize/60)
        #print self.noisefilt[0]
        #print power[0]
        #noise_th = np.mean(10*np.log(np.abs(spectrum)))/5 #*self.demodFiltSize/10
#        gain = np.greater((self.demodFiltSize/10)*pwr,self.noisefilt)
#        windowsize=256 #2*self.demodFiltSize
#       window = np.ones(windowsize)
#        spectrum *= signal.fftconvolve(gain,window,mode='same')
#        output = np.fft.ifft(spectrum)

 #       print output[0]
        #wvlt = signal.ricker
        #widths = np.arange(1,11)
        # wvltxfm =
###### Wavelet Denoising ##########
   #     wvltxfm = pywt.wavedec(output, 'db20', level=8)
    #    filtwvlt = []
     #   for lvl in wvltxfm:
      #      filtwvlt.append( lvl*np.greater(lvl,(self.demodFiltSize/60.0)))
       # output = pywt.waverec(filtwvlt,'db20')


        # deemphasis
        b,a = signal.butter(1, 2122/48000*2*np.pi, btype='low')
        output, zf = signal.lfilter(b, a, output,zi=self.demph_zf)
        self.demph_zf = zf

        stereo = np.zeros(output.size*2, dtype='complex')
        if (isStereo):

            diff = signal.lfilter(b,a,diff)
            w = self.stereoWidth  # adjust to change stereo wideness
         #print w
            left = output + w/10 * diff
            right = output - w/10 * diff

            if(self.useAudioFilter):
                left = self.lowpass(left,self.audioFilterSize)
                right = self.lowpass(right,self.audioFilterSize)

            stereo[0:stereo.size:2] = left
            stereo[1:stereo.size:2] = right
        else:
            if self.useAudioFilter:
                output = self.lowpass(output,self.audioFilterSize) # just the tip (kills the 19k pilot)
            stereo[0:stereo.size:2] = output
            stereo[1:stereo.size:2] = output


        #spectrum = np.fft.fft(stereo[::2])
        spectrum = np.fft.fft(stereo[::2]*spectral_window(output.size))
        self.plspectrogram = np.roll(self.plspectrogram, 1,axis=1)
        self.plspectrogram[:,0] = np.log(np.abs(spectrum[spectrum.size/2:spectrum.size:20]))
        if(self.toDraw and self.plotPlaying): # and self.count % 2 == 0):
            if self.toDrawWaterfalls:
                self.drawPlspectrum()
            else:
                sm = np.abs(np.fft.fftshift(spectrum[::20]))
                toPlot = (np.linspace(-2.4e4,2.4e4,sm.size),sm)
                self.replot(toPlot)


        if(self.toDraw and self.plotWaveform):
            sm = np.real(output[::20])
            toPlot = (np.linspace(0,output.size/48000,sm.size),sm)
            self.replot(toPlot)


        return np.real(stereo)


    # Alternate demodulator. Not used, but extremely simple
    def demodulate2(self,samples):
        # DEMODULATION CODE

        # LIMITER goes here

        # low pass & down sampling
        h = signal.firwin(128,80000,nyq=1.2e5)
        lp_samples = signal.fftconvolve(samples, h)

    # polar discriminator

        A = lp_samples[1:lp_samples.size]
        B = lp_samples[0:lp_samples.size-1]

        dphase = ( A * np.conj(B) ) / np.pi

        dphase.resize(dphase.size+1)
        dphase[dphase.size-1] = dphase[dphase.size-2]

        h = signal.firwin(128,16000,nyq=1.2e5)
        rebuilt = signal.fftconvolve(dphase,h)

        output = rebuilt[::self.decim_r2]

        output = self.lowpass(output, self.audioFilterSize)

        return np.real(output)

# utility functions #

    def lowpass(self,x,width):
        #wndw = np.sinc(np.r_[-15:16]/np.pi)/np.pi
        #wndw = np.kaiser(width,6)
        wndw = signal.firwin(16,width*999,nyq=24000)
        #wndw /= np.sum(wndw)
        new_array = signal.fftconvolve(x, wndw, mode='same')

        return new_array

    # calculate mean average deviation #
    def mad(self,samples):
        ave = np.mean(samples)
        return np.mean(np.abs(samples-ave))

    # calculate rms for power #
    def rms(self,samples):
        meansq = np.mean(np.square(samples))
        return np.sqrt(meansq)

    def play(self,samples):
        self.stream.write( samples.astype(np.float32).tostring() )


    # starting point
    def start(self):
        # Initiates running things
        self.streamer = MakeDaemon(self.demodulate_th) # run demodulation in the 'background'
        self.streamer.start()
        self.count = 0
        self.sampler_t = threading.Thread(target=self.getSamplesAsync) # sampler loop
        self.sampler_t.start()



class MakeDaemon(threading.Thread):

    def __init__(self, function, args=None):
        threading.Thread.__init__(self)
        self.runnable = function
        self.args = args
        self.daemon = True

    def run(self):
        self.runnable()

class PhaseLockedLoop():
    phase = 0
    #phase_shift = 0
    beta = .1
    def __init__(self,size,freq,samplerate):
        self.pll = np.exp(2j*np.pi*freq/samplerate*np.r_[0:size])
        self.size = size
        self.freq=freq
        self.samplerate = samplerate
    def adjust(self,pilot):
        #mul = pilot * self.pll
        dphase = np.angle( self.pll * np.conj(pilot))
        self.phase += self.beta * dphase
        self.pll = np.exp(2j*np.pi*self.freq/self.samplerate*(np.r_[0:self.size] - self.phase))
    def pllx2(self):
        return np.real(np.square(self.pll))



def main():

    app = QtGui.QApplication(sys.argv)

    freq = 90.7e6
    radio = FMRadio(freq,32768)
    print "Currently listening to: ",
    print freq/1e6,
    print "MHz"

    radio.show()

    radio.start()

    app.exec_()
    radio.__del__()


# Run the program
main()

