

from __future__ import division
import numpy as np

from scipy import signal
from scipy.signal import signaltools as sigtool
from rtlsdr import RtlSdr
import pyaudio

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import animation
from matplotlib import ticker

#from multiprocessing.pool import Process,ThreadPool
import threading
import Queue
import time

from PyQt4 import QtCore, QtGui
from PyQt4.Qwt5 import QwtPlotCurve,QwtPlot
from radioui import Ui_MainWindow
import sys

#import cv2
#img = np.zeros((200,200))
#cv2.imshow('spectrum',img)


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
    useLPFilt = False
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

    toPlot = (np.cumsum(np.ones(5780)),np.cumsum(np.ones(5780)))

    def __init__(self,freq,N_samples):

        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.createQtConnections()


        self.sample_rate = 2.4e5 ###1e6
        #self.decim_r1 = 1e6/2e5 # for wideband fm
        self.decim_r2 = 2.4e5/48000 # for baseband recovery
        self.center_freq = freq #+250e3
        self.gain = 38

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

        #self.anim = animation.FuncAnimation(self.fig,self.replot,interval=2000,blit=False)
        #self.canvas.mpl_connect('click_event',self.on_click_plot)


#         self.curve = QwtPlotCurve("Frequencies")
#         self.curve.setData(self.toPlot[0],self.toPlot[1]) #np.r_[-5e5:5e5:1e6/self.toPlot.size],np.abs(self.toPlot))
#         self.curve.attach(self.ui.spectrumPlot)
#         self.ui.spectrumPlot.replot()
#         #threading.Timer(1,self.replot).start()

    def initplot(self):
        self.axes = self.fig.add_subplot(111, aspect=200/431)
        self.axes.xaxis.set_major_locator(ticker.NullLocator())
        self.axes.yaxis.set_major_locator(ticker.NullLocator())
        #self.axes.invert_yaxis()

    def replot(self):
       self.axes.clear()
       #self.axes.set_s
       self.axes.plot(self.toPlot[0],self.toPlot[1])
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
        freq /= 10.0
        text = "%.1f MHz" % freq
        self.ui.curFreq.setText(text)
        self.center_freq = freq*1e6 #+ 250e3
        setf_t = threading.Thread(target=self.setF_th, args=[self.center_freq,])
        setf_t.start()
        setf_t.join()

    def setF_th(self,f):
        while(self.is_sampling == True):
            pass
        self.sdr.center_freq = f

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
        if(s % 2 == 0):
            s+=1
        self.demodFiltSize = s

    def setAudioFiltUse(self,s):
        self.useAudioFilter = s
    def setAudioFiltSize(self,s):
        self.audioFilterSize = s

    def terminate(self):
        self.__del__()

    def __del__(self):
        #QtGui.QMainWindow.__del__()
        #Ui_MainWindow.__del__()
        #self.streamer.stop()
        #self.sampler_t.stop()
        print "sdr closed"
        self.sdr.close()
        print "pyaudio terminated"
        self.pa.terminate()
        sys.exit()

    def getSamples(self):
        #N_samples = self.N_samples # 1/24.4 seconds   ~46336 #approximately a 2048/44100 amount's time
        return self.sdr.read_samples(self.N_samples)

    def getSamplesAsync(self):
        #Asynchronous call. Meant to be put in a loop w/ a calback fn
        #print 'gonna sample'
        self.is_sampling = True
        samples = self.sdr.read_samples_async(self.sampleCallback,self.N_samples,context=self)

    def sampleCallback(self,samples,sself):

        self.is_sampling = False
        self.sample_buffer.put(samples)
        #print 'put some samples in the jar'

        # recursive loop
        #sself.getSamplesAsync()

    def demodulate_th(self):
        while(1):

            try:
                samples = self.sample_buffer.get()
                #samples2 = self.sample_buffer.get()
              #  print 'gottum'
            except:
                print "wtf idk no samples?"
                break

            out1 = self.demodulate(samples)
            self.sample_buffer.task_done()
            #out2 = self.demodulate(samples2)
            #self.sample_buffer.task_done()

            audio_out = out1 #np.append(out1,out2)

            self.play(audio_out)

        print 'gonna try to finish off the to-do list'
        sample_buffer.join()


    def demodulate(self,samples):
        # DEMODULATION CODE
        #samples = #self.sample_buffer.get()
        # LIMITER goes here

        # low pass & down sampling via fft


        spectrum = np.fft.fftshift(np.fft.fft(samples))

        self.spectrogram = np.roll(self.spectrogram, 1,axis=1)
        self.spectrogram[:,0] = np.log(np.abs(spectrum[::100]))


        if(self.toDraw and self.plotOverall and self.count % 10 == 9):

#             self.toPlot = (np.linspace(-5e5,5e5,spectrum.size),np.abs(spectrum))
#             self.replot()
            self.drawSpectrum()

        self.count += 1



            #fig = plt.figure()
            #plt.plot(np.abs(spectrum))
            #plt.show()
#
#         spectrum *= self.lpf
# # Decimate in two rounds. One to 200k, another to 44.1k
#         # DECIMATE HERE. Note that we're looking at 1MHz bandwidth.
#         n_s = spectrum.size
#
#         channel_spectrum = spectrum #.25*spectrum[int(n_s*.75-.5*n_s/self.decim_r1):int(.75*n_s+.5*n_s/self.decim_r1)] #np.append(spectrum[0:int(n_s/self.decim_r1*.5)],spectrum[n_s-int(n_s/self.decim_r1*.5):n_s])
#
#         #radio_spectrum -= np.mean(radio_spectrum) #attempt to remove dc bias
#         #print channel_spectrum.size

# #             fig = plt.figure()
# #             plt.plot(np.abs(np.fft.ifftshift(channel_spectrum)))
# #             plt.show()
#             #self.fig.plot(np.fft.ifftshift(channel_spectrum))
#
#
#         lp_samples = np.fft.ifft(np.fft.ifftshift(channel_spectrum))
#
        lp_samples = samples



      #  lp_samples /= power


    # polar discriminator



        dphase = np.zeros(lp_samples.size, dtype='complex')

        A = lp_samples[1:lp_samples.size]
        B = lp_samples[0:lp_samples.size-1]

        dphase[1:] = ( A * np.conj(B) )

        dphase[0] = lp_samples[0] * np.conj(self.prevCutoff) #dphase[dphase.size-2]
        self.prevCutoff = lp_samples[lp_samples.size-1]

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


        spectrum = np.fft.fft(rebuilt)


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

            #stereo_spectrum = spectrum
            if self.useStereo:
                isStereo = True
                #modulated = rebuilt * np.exp(-2j*np.pi*38000/2.4e5*np.r_[0:rebuilt.size])
                #h = signal.firwin(128,22000,nyq=1.2e5)
                #lp_mod = signal.fftconvolve(modulated,h,mode='same')
                #decim = lp_mod[::self.decim_r2]
                #dphase = np.zeros(decim.size, dtype='complex')
#
#                A = decim[1:decim.size]
#                B = decim[0:decim.size-1]
#
#                dphase[1:] = np.angle( A * np.conj(B) )
                #h = signal.firwin(128,22000,nyq=24000)

                #diff = dphase# [::self.decim_r2]
                mod_spectrum = np.roll(spectrum,int(.5*n_z*38000/120000))
                mod_spectrum += np.roll(spectrum,int(-.5*n_z*38000/120000)) #np.fft.fft(modulated)
                mod_spectrum *= self.lpf #np.fft.ifftshift(np.hamming(stereo_spectrum.size))
                stereo_spectrum = np.append(mod_spectrum[0:np.ceil(n_z/self.decim_r2*.5)],mod_spectrum[n_z-np.ceil(n_z/self.decim_r2*.5):n_z])
                diff = np.fft.ifft(stereo_spectrum)
            #self.base_spectrum = np.append(spectrum[0:int(n_z/self.decim_r2*.5)],spectrum[n_z-int(n_z/self.decim_r2*.5):n_z])
            #output = np.fft.ifft(self.base_spectrum)
            h = signal.firwin(128,16000,nyq=1.2e5)
            output = signal.fftconvolve(rebuilt,h,mode='same')
            output = rebuilt[::self.decim_r2]
#             h = signal.firwin(128,16000,nyq=2.4e4)
#             output = signal.fftconvolve(output,h,mode='same')
        elif demodSub1:

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

        output -= np.mean(output)

        stereo = np.zeros(output.size*2, dtype='complex')
        if (isStereo):
            #diff = np.fft.ifft(stereo_spectrum)
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


        spectrum = np.fft.fft(stereo[::2])
        self.plspectrogram = np.roll(self.plspectrogram, 1,axis=1)
        self.plspectrogram[:,0] = np.log(np.abs(spectrum[spectrum.size/2:spectrum.size:20]))
        if(self.toDraw and self.plotPlaying): # and self.count % 2 == 0):
            if self.toDrawWaterfalls:
                self.drawPlspectrum()
            else:
                sm = np.abs(np.fft.fftshift(spectrum[::20]))
                self.toPlot = (np.linspace(-2.4e4,2.4e4,sm.size),sm)
                self.replot()


        if(self.toDraw and self.plotWaveform):
            sm = np.real(stereo[::20])
            self.toPlot = (np.linspace(0,output.size/48000,sm.size),sm)
            self.replot()


        return np.real(stereo)


#     def updateimg(self,base_spectrum):
#         spectralimg = np.zeros((base_spectrum.size/20,base_spectrum.size/10))
#         for i in np.r_[0:base_spectrum.size/10]:
#             spectralimg[np.abs(np.fft.fftshift(base_spectrum)[i*10]/10),i] = 1
#         cv2.imshow('spectrum',spectralimg)

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

        return np.real(.5*output)

# utility functions #

    def lowpass(self,x,width):
        #wndw = np.sinc(np.r_[-15:16]/np.pi)/np.pi
        #wndw = np.kaiser(width,6)
        wndw = signal.firwin(16,width*990,nyq=24000)
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



    def start(self):
        self.streamer = MakeDaemon(self.demodulate_th) # run demodulation in the 'background'
        self.streamer.start()
        self.count = 0
        self.sampler_t = threading.Thread(target=self.getSamplesAsync) # sampler loop
        self.sampler_t.start()
        #while(1):
        #    time.sleep(1)
        #    print('replot')
        #    self.replot()


class MakeDaemon(threading.Thread):

    def __init__(self, function, args=None):
        threading.Thread.__init__(self)
        self.runnable = function
        self.args = args
        self.daemon = True

    def run(self):
        self.runnable()






    # the following are for the Qt UI setup







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

main()

