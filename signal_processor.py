########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from multiprocessing import current_process, Event, Process, Pipe
from threading import Thread, Condition
import time
# from itertools import filterfalse

import face_recognition
import psutil
import numpy as np
import mediapipe as mp
import cv2
import scipy
from scipy.signal import sosfiltfilt, butter, find_peaks
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, IncrementalPCA, NMF
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use("TKAgg", force=True)

from packets import FPtoSP_Packet, SPtoIR_Packet
from frame_processor import MAX_ENCODINGS, MAX_NUM_FACES
from util import interp_peak

POS_FRAMES  = 48    # uses RGB
HR_FRAMES   = 360   # uses PPG
BR_FRAMES   = 599   # uses PPG
O2_FRAMES   = 361   # uses RGB
STRIDE      = 30

MIN_HR_HZ   = 0.65
MAX_HR_HZ   = 3

MIN_HR      = 60*MIN_HR_HZ
MAX_HR      = 60*MAX_HR_HZ

T_LAMBDA    = 10


E_K         = np.array([1,   0,0,0,0,0,0,0,0,0])
E           = np.array([0.1 ,1  ,0.1    ,0,0,0,0,0,0,0,0,0])
E_ALT       = np.array([1   ,1  ,1      ,0,0,0,0,0,0,0,0,0])
E_ALT2      = np.array([0.1 ,1  ,0.15   ,0,0,0,0,0,0,0,0,0])

FILTER_MC=1
FILTER_DMC=1
FILTER_BETA=0.001

### Testings
INVERSE_TYPE = None
DISPLAY      = None
# TARVAIN_RGB  = True
# TARVAIN_NOISE= True
NOISE_SCALE  = False
POS_SCALE    = False
PRINT        = False
DIRECT_HR    = False
STACKED_NOISE= False

def create_signal_processor(debug_lock, SP_FPtoSP, **kwargs):
    
    if "SignalProcessor" in [p.name for p in psutil.process_iter()]:
        raise RuntimeError("SignalProcessor instance already running")
    
    run_event = Event()
    SP_SPtoIR, IR_SPtoIR = Pipe()
    
    process = Process(target=_SignalProcessor.process_start_point, args=[debug_lock, SP_FPtoSP, SP_SPtoIR, run_event],kwargs=kwargs, name="SignalProcessor", daemon=True)
    
    process.start()
    
    return process, run_event, IR_SPtoIR

class _SignalProcessor():
    @staticmethod
    def process_start_point(debug_lock, SP_FPtoSP, SP_SPtoIR, run_event, **kwargs):
        instance = _SignalProcessor(debug_lock, SP_FPtoSP, SP_SPtoIR, run_event, **kwargs)
    
    def _print(self, string):
        self.debug_lock.acquire()
        try:
            print("{0: <16}: ".format(self.proc_name) + str(string), flush=True)
        finally:
            self.debug_lock.release()
    
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    EPS = 1e-9
    
    def __init__(self, debug_lock, SP_FPtoSP, SP_SPtoIR, run_event, **kwargs):
        self.proc_name = current_process().name
        if  self.proc_name != 'SignalProcessor':
            raise RuntimeError("SignalProcessor must be instantiated on a seperate process, please use 'create_frame_processor' instead")
        
        
        self.debug_lock = debug_lock
        self.SP_FPtoSP = SP_FPtoSP
        self.SP_SPtoIR = SP_SPtoIR
        self.run_event = run_event
        
        
        self.sig_len            = 300
        self.sig_circ_len       = self.sig_len*2
        self.signal             = np.zeros((MAX_ENCODINGS, self.sig_circ_len,12),dtype=np.float64)
        self.presence           = np.zeros((MAX_ENCODINGS, self.sig_circ_len), dtype=bool)
        self.sig_idx            = 0
        
        self.in_times           = np.zeros(self.sig_circ_len, dtype=np.int64)
        
        self.running_counts     = np.zeros(MAX_ENCODINGS, np.uint64)
        self.missing_counts     = np.zeros(MAX_ENCODINGS, np.uint8)
        self.prev_label_mask    = np.zeros(MAX_ENCODINGS, dtype=bool)
        
        
        self.pre_filt           = butter(5, [MIN_HR_HZ, MAX_HR_HZ], fs=30.0000003, output='sos', btype='bandpass')#= butter(10, MAX_HR_HZ*2, fs=30, output='sos', btype='lowpass')
        self.HR_filt            = butter(10, [MIN_HR_HZ, MAX_HR_HZ], fs=30.0000003, output='sos', btype='bandpass')
        self.full_label         = np.arange(MAX_ENCODINGS)
        
        self.RGB_slice          = np.s_[:,:,:3]
        # self.noise_slice        = np.s_[:,3:]
        
        self.POS_len            = self.sig_len
        self.POS_circ_len       = self.sig_len*2
        self.POS_sig            = np.zeros((MAX_ENCODINGS,self.POS_circ_len),dtype=np.float64)
        self.POS_circ_idxs      = np.arange(POS_FRAMES, dtype=np.int32)
        
        self.stacked_noise      = np.zeros((MAX_ENCODINGS, self.POS_circ_len,9),dtype=np.float64)
        
        self.HR_ready           = np.zeros(MAX_ENCODINGS,dtype=bool)
        self.HR_len             = 16
        # self.HR_circ_len        = self.HR_len*2
        self.HRs                = np.zeros((MAX_ENCODINGS,self.HR_len))
        self.HR_weights         = np.zeros((MAX_ENCODINGS,self.HR_len))
        self.HR_idxs            = np.zeros((MAX_ENCODINGS), dtype=np.uint8)
        
        ### OEF
        self.output_HR          = np.zeros(MAX_ENCODINGS)
        self.filter_prev_t      = np.zeros((MAX_ENCODINGS), dtype=np.float64)
        self.filter_dHR         = np.zeros(MAX_ENCODINGS)
        
        self.signal_cond        = Condition()
        self.thread             = Thread(target=self._process_loop, name="SignalProcessor_proc_thread", daemon=True)
        
        self.null_sent          = False
        # self.prev_packet        = FPtoSP_Packet(MAX_ENCODINGS)
        
        self.run_event.wait()
        self.tmp_last_time = 0
        self.debug_flag = True
        
        # self.models = [IncrementalPCA(whiten=True) for _ in range(MAX_ENCODINGS)]
        
        if DISPLAY is not None:
            self.debug_thread = Thread(target=self.sebud, name="SignalProcessor_sebud_thread", daemon=True)
            self.debug_thread.start()
        
        self.tmp_idx = 0
        self._input_loop()
        
        
    def sebud(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not DISPLAY.startswith('corr'):
            line0, = ax.plot([], [],'k',linewidth=2.5,label="POS")
            #std
                # 4  dX
                # 5  dY
                # 6  dSize
                # 7  absdiff mean
                # 8  diff std
                # 9  outside R
                # 10 outside G
                # 11 outside B
            line1, = ax.plot([], [],label="Region STD")
            line2, = ax.plot([], [],label="X Movement")
            line3, = ax.plot([], [],label="Y Movement")
            line4, = ax.plot([], [],label="Size Change")
            line5, = ax.plot([], [],label="Region Diff Mag")
            line6, = ax.plot([], [],label="Region Diff STD")
            line7, = ax.plot([], [],label="Background R")
            line8, = ax.plot([], [],label="Background G")
            line9, = ax.plot([], [],label="Background B")
            # x = np.arange(POS_FRAMES)
            self.temp_x = np.arange(self.sig_len)
            self.temp_ys = [np.zeros(self.sig_len) for _ in range(10)]
            
            for i, line in enumerate([line0,line1,line2,line3,line4,line5,line6,line7,line8,line9]):
                line.set_xdata(self.temp_x)
                line.set_ydata(self.temp_ys[i])
            
            plt.legend()
            
        else:
            cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            bars = ax.bar([1,2,3,4,5,6,7,8,9],[0]*9,color=cols[:9])
            ax.set_xticks([1,2,3,4,5,6,7,8,9], ["Region STD", "X Movement", "Y Movement", "Size Change", "Region Diff Mag", "Region Diff STD", "Background R", "Background G", "Background B"])
            ax.set_title("POS correlation with noise signals")
            ax.set_ylim([-1,1])
            self.temp_ys = np.zeros(9)
        
        def temp_f(i):
            for i, line in enumerate([line0,line1,line2,line3,line4,line5,line6,line7,line8,line9]):
                line.set_xdata(self.temp_x)
                line.set_ydata(self.temp_ys[i])
            ax.relim()
            ax.autoscale_view()
            return line0, line1,line2,line3,line4,line5,line6,line7,line8,line9
            
        def temp_b(i):
            for i, b in enumerate(bars):
                b.set_height(self.temp_ys[i])
        
        if DISPLAY.startswith('corr'):
            ani = FuncAnimation(fig, temp_b, interval=1000, save_count=0)
        else:
            ani = FuncAnimation(fig, temp_f, interval=30, blit=True, save_count=0)
        
        plt.show()
    
    def _spect_ratio(self, x):
        x = x-np.mean(x,axis=-1,keepdims=True)
        x = x * np.hanning(x.shape[-1])
        fft = np.fft.rfft(x, n=8 * x.shape[-1])
        spectrum = np.abs(fft)
        freqs = np.linspace(0, 15 * 60, spectrum.shape[-1])
        idx = np.where((freqs >= MIN_HR) & (freqs <= MAX_HR))
        ratio = np.sum(spectrum[:,idx[0]],axis=-1)/np.sum(spectrum,axis=-1)
        return ratio
    
    def _to_spect(self, x):
        x = x * np.hanning(x.shape[-1])
        fft = np.fft.rfft(x, n=8 * x.shape[-1])
        spectrum = np.abs(fft)
        freqs = np.linspace(0, 15 * 60, spectrum.shape[-1])
        idx = np.where((freqs >= MIN_HR) & (freqs <= MAX_HR))
        freqs = freqs[idx]
        if spectrum.ndim>1:
            spectrum = spectrum[:,idx[0]]
        else:
            spectrum = spectrum[idx]
        return freqs, spectrum
    
    def _OEF(self, label, new_HR):
        
        HR                 = self.output_HR[label]
        prev_t             = self.filter_prev_t[label]
        dHR                = self.filter_dHR[label]
        t                  = time.perf_counter()
        if prev_t == 0:
            prev_t = t
            x_hat = new_HR
            dHR = 0
        else:
            t_e = t - prev_t
            r = 2 * np.pi * FILTER_DMC * t_e
            a_d = r / (r + 1)
            dx = (HR - new_HR) / t_e
            dHR = a_d * dx + (1 - a_d) * dHR
            
            cutoff = FILTER_MC + FILTER_BETA * np.abs(dHR)
            r = 2 * np.pi * FILTER_MC * t_e
            a = r / (r + 1)
            x_hat = a * HR + (1 - a) * new_HR
            
        self.output_HR[label] = x_hat
        self.filter_dHR[label] = dHR
        self.filter_prev_t[label] = t
    
    def _tarvainen(self, x):
        t = x.shape[1]
        l = t/T_LAMBDA
        I = np.identity(t)
        D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t-2, t)).toarray()
        Hinv = np.linalg.inv(I+l**2*(np.transpose(D2).dot(D2)))
        for i in range(len(x)):
            detrendedX = (I - Hinv).dot(x[i])
            x[i] = detrendedX
        
    def _process_loop(self):
        
        # plt.ion()
        # model = IncrementalPCA(whiten=True)
        while self.run_event.is_set():
            with self.signal_cond:
                if not self.signal_cond.wait_for(lambda: np.any(self.HR_ready), timeout=3):
                    continue
                packet = SPtoIR_Packet()
                labels      = self.full_label[self.HR_ready]
                # rgbs = self.signal[self.HR_ready, self.sig_idx:self.sig_idx+self.sig_len, :3].copy()
                if STACKED_NOISE:
                    noise_signals = self.stacked_noise[self.HR_ready, self.POS_circ_idxs[-1]:self.POS_circ_idxs[-1]+self.sig_len].copy()
                else:
                    noise_signals = self.signal[self.HR_ready, self.sig_idx:self.sig_idx+self.sig_len, 3:].copy()#.copy()
                pos_signals = self.POS_sig[self.HR_ready, self.POS_circ_idxs[-1]:self.POS_circ_idxs[-1]+self.sig_len].copy()#.copy()
                # self._print(self.presence.shape)
                pres        = self.presence[self.HR_ready, self.sig_idx:self.sig_idx+self.sig_len].copy()
                # self._print(pres.shape)
                # print(all_signals.shape)
                # print(pos_signals.shape)
                #k = np.dstack([pos_signals, all_signals[...,3:]])
                #k2 = np.dstack([pos_signals, all_signals])
                timeso = self.in_times[self.sig_idx:self.sig_idx+self.sig_len].copy()
                print(f"{self.sig_idx}:{self.sig_idx+self.sig_len}")
                # self._print(f"All: {all_signals.shape}")
                # Faces, Frames, Channels
                self.HR_ready[:] = False
            
            # self._tarvainen(noise_signals)
            # self._tarvainen(rgbs)
            # self._tarvainen(all_signals[self.RGB_slice])
            # # self._tarvainen(all_signals[:,:,-3:])
            # k= (k-np.mean(k,axis=1,keepdims=True))/np.std(k,axis=1,keepdims=True)
            
            # model = NMF()
            # k2= (k2-np.mean(k2,axis=1,keepdims=True))/np.std(k2,axis=1,keepdims=True)
            # all_signals[...,3:] = (all_signals[...,3:]-np.mean(all_signals[...,3:],axis=1,keepdims=True))/np.std(all_signals[...,3:],axis=1,keepdims=True)
            # all_signals[...,3:] = sosfiltfilt(self.pre_filt, all_signals[...,3:], axis=1)
            
            # # all_signals = sosfiltfilt(self.pre_filt, all_signals, axis=1)
            
            # k = sosfiltfilt(self.pre_filt, k, axis=1)
            
            # model = PCA(whiten=True)#FastICA(whiten='unit_variance')
            
            # 0  R
            # 1  G
            # 2  B
            # 3  std
            # 4  dX
            # 5  dY
            # 6  dSize
            # 7  absdiff mean
            # 8  diff std
            # 9  outside R
            # 10 outside G
            # 11 outside B
            
            for i,l in enumerate(labels):
                pos = pos_signals[i,pres[i]]
                # rgb = rgbs[i,pres[i]]
                noise = noise_signals[i,pres[i]]
                k = np.hstack([pos[:,None], noise])
                times   =  timeso[pres[i]]
                # print(times)
                # print(self.tmp_last_time)
                try:
                    spl = CubicSpline(times,k,axis=0,bc_type='natural')
                except Exception as e:
                    print(np.diff(times))
                    raise e
                x = np.arange(times[0], times[-1], 33333333, dtype=np.uint64)
                k = spl(x)
                
                if NOISE_SCALE: 
                    k[:,1:] = (k[:,1:]-np.mean(k[:,1:],axis=0,keepdims=True))/np.std(k[:,1:],axis=0,keepdims=True)#/np.max(np.std(k[:,1:],axis=0,keepdims=True),axis=1,keepdims=True) #* np.std(k[:,1:],axis=None,keepdims=True)
                    # B = np.sum(np.abs(k[:,1:]),axis=0,keepdims=True)
                    # A = np.mean(B,axis=1,keepdims=True)
                    # k[:,1:] = k[:,1:]/B * A
                if POS_SCALE: k[:,0] = (k[:,0]-np.mean(k[:,0]))/np.std(k[:,0])
                
                
                signals = k#[i]
                
                # ratios = self._spect_ratio(signals[:,1:].T)
                
                if DISPLAY in ['corr_normal', 'corr']:
                    corrs = np.corrcoef(signals.T)
                    self.temp_ys = corrs[0,1:]
                
                    
                # signals = sosfiltfilt(self.HR_filt, signals, axis=0)
                # signals = signals.T
                # signals = model.fit_transform(signals)
                # print(signals.shape)
                
                        
                # line1.set_xdata(x)
                # line1.set_ydata(signals[...,3])
                # line2.set_xdata(x)
                # line2.set_ydata(signals[...,4])
                # line3.set_xdata(x)
                # line3.set_ydata(signals[...,5])
                # line4.set_xdata(x)
                # line4.set_ydata(signals[...,6])
                # line5.set_xdata(x)
                # line5.set_ydata(signals[...,7])
                # line6.set_xdata(x)
                # line6.set_ydata(signals[...,8])
                # line7.set_xdata(x)
                # line7.set_ydata(signals[...,9])
                # line8.set_xdata(x)
                # line8.set_ydata(signals[...,10])
                # line9.set_xdata(x)
                # line9.set_ydata(signals[...,11])
                
                # # # self._print(f"Single: {signals.shape}")
                # # # Frames, Channels
                # # # signals[:,3:] = sosfiltfilt(self.HR_filt, signals[:,3:], axis=0)
                # # # signals = signals[40:-40]
                # # # mag = np.mean(np.abs(signals[:,3:]))
                # # # self._print(mag)
                
                # print(np.sum(np.abs(signals[3:])))
                # signals = sosfiltfilt(self.HR_filt, signals, axis=0)
                # signals[:,1:] = model.fit_transform(signals[:,1:])
                # signals[:,3:] = model.fit_transform(signals[:,3:])
                
                # signals[:,3:] = model.fit_transform(signals[:,3:])
                
                # self.models[l].partial_fit(signals[:,3:])
                # signals[:,3:] = self.models[l].transform(signals[:,3:])
                # signals[:,3:] = signals[:,3:] * stds
                
                # # # signals[:,3:] = signals[:,3:]*mag
                # signals = signals.T
                # signals = sosfiltfilt(self.HR_filt, signals, axis=1)
                # m = (np.linalg.pinv(signals@signals.T)@signals)
                # for z in range(10):
                    # self.temp_ys[z] = m[...,z]
                # print(m.shape)
                # self._print(signals.shape)
                
                if INVERSE_TYPE in ['after','both']:
                    # model.partial_fit(signals[:,1:].T)
                    model  = PCA()
                    model.fit(signals[:,1:].T)
                    signals[:,1:] = model.components_.T# * model.explained_variance_[:,None]).T
                    # signals[:,1:] = model.transform(signals[:,1:])
                    # signals = signals.T
                    signals = (np.linalg.pinv(signals@signals.T)@signals)
                    # # signals = signals.T
                    # signal = signals[...,0]
                    # #m[...,0] ##signals[0]#.T
                # else:
                    # signal = signals[...,0]
                    
                # m = (np.linalg.pinv(signals@signals.T)@signals)
                # signal = m[...,0]
                if DISPLAY=='normal':   
                    self.temp_x = x * 1e-9
                    for z in range(10):
                        self.temp_ys[z] = signals[...,z]
                
                signals = sosfiltfilt(self.HR_filt, signals, axis=0)
                
                if DISPLAY == 'corr_filt':
                    corrs = np.corrcoef(signals.T)
                    self.temp_ys = corrs[0,1:]
                
                if DISPLAY=='filt':
                    self.temp_x = x * 1e-9
                    for z in range(10):
                        self.temp_ys[z] = signals[...,z]
                
                # signals = signals.T
                # Channels, Frames
                # signal = E @ np.linalg.pinv(signals@signals.T)@signals
                # signal = sosfiltfilt(self.HR_filt, signal, axis=0)
                # # # signal = signal[30:-30]
                
                # self._print(f"Extracted: {signal.shape}")
                freq, spectra = self._to_spect(signals.T)
                spectrum = spectra[0]
                if DISPLAY=="spectra":
                    
                    self.temp_x = freq
                    for z in range(10):
                        self.temp_ys[z] = spectra[z]
                        
                    # self.temp_ys[0] = spectra[0]/np.max(spectra[0])
                    # self.temp_ys[1] = np.sum(spectra[1:],axis=0)/np.max(np.sum(spectra[1:],axis=0))
                    # self.temp_ys[2] = np.prod(spectra[1:],axis=0)/np.max(np.prod(spectra[1:],axis=0))
                    # self.temp_ys[3] = np.max(spectra[1:],axis=0)/np.max(np.max(spectra[1:],axis=0))
                    
                    # for z in range(4,10):
                        # self.temp_ys[z] = np.zeros(spectra.shape[-1])
                    
                # line.set_xdata(np.arange(signals.shape[1]))
                # line.set_ydata(signals[1])
                # line1.set_xdata(freq)
                # line1.set_ydata(spectrum)
                # plt.draw()
                # ax.relim()
                # ax.autoscale_view()
                # fig.canvas.draw()
                # fig.canvas.flush_events()
                hr = interp_peak(freq, spectrum)
                peak_indices, peak_dict = find_peaks(spectrum, height=np.max(spectrum)/10)
                if len(peak_indices)==1:
                    weight = 1
                elif len(peak_indices)==0:
                    weight = 0
                else:
                    peak_heights = peak_dict['peak_heights']
                    highest_peak = spectrum[peak_indices[np.argmax(peak_heights)]]
                    second_peak = spectrum[peak_indices[np.argpartition(peak_heights,-2)[-2]]]
                    weight = 1 - second_peak/highest_peak
                # self._print(hr)
                w1 = weight
                weight = weight/np.sum(spectra[1:]) #/ np.sum(ratios)#* np.sqrt(np.sum(spectra[0])/np.sum(spectra[1:]))
                if DIRECT_HR:
                    self.HRs[l,self.HR_idxs[l]] = hr
                    self.output_HR[l] = hr
                    if l==0 and PRINT: self._print(f"{str(self.tmp_idx):<3},{hr:<6.2f}")
                else:
                    self.HRs[l,self.HR_idxs[l]] = hr
                    self.HR_weights[l, self.HR_idxs[l]]= weight +self.EPS#/ np.sum(np.abs(signals[1:])) 
                    self.HR_idxs[l] = (self.HR_idxs[l]+1)%self.HR_len
                    weighted_hr = np.average(self.HRs[l],weights=self.HR_weights[l],axis=-1)
                    self._OEF(l, weighted_hr)
                    if l==0 and PRINT: self._print(f"{str(self.tmp_idx):<3}, {hr:<6.2f}, {w1:<6.3f}, {1/np.sum(spectra[1:]):<6.3f}, {weight:<6.3f}, {weighted_hr:<6.2f}, {self.output_HR[l]:<6.2f}")
                self.tmp_idx+=1
                # freq, spectrum = self._to_spect(all_signals[0,:,4])
                # line.set_xdata(np.arange(signals.shape[1]))
                # line.set_ydata(signals[1])
                # line2.set_xdata(freq)
                # line2.set_ydata(spectrum)
                
            # weights = self.HR_weights[labels].copy()
            # self._print(self.HRs[self.HR_idxs>0].shape)
            # self._print((self.HRs[self.HR_idxs>0]>0).shape)
            # weights[self.HR_idxs[labels]>0] = weights[self.HR_idxs[labels]>0] / np.minimum(np.abs(self.HRs[labels & self.HR_idxs>0]-np.average(self.HRs[labels & self.HR_idxs>0],weights=(self.HRs[labels & self.HR_idxs>0]>0),axis=-1,keepdims=True)),1)
            # print(self.HRs[labels].shape)
            # print(weights[labels].shape)
            # packet.HR_vals = np.average(self.HRs[labels],weights=self.HR_weights[labels],axis=-1)
            
            packet.labels = labels
            
            packet.HR_vals = self.output_HR[labels]
            self.SP_SPtoIR.send(packet)
                
        
    def _input_loop(self):
        self.running = True
        count = 0
        packet_wait_time = 0
        lock_time = 0
        loop_time = 0
        exec_time = 0
        
        first_packet = True
        self.thread.start()
        
        while self.run_event.is_set():
            t0 = time.perf_counter()
            packet = self.SP_FPtoSP.recv()
            if not first_packet:
                packet_wait_time += (time.perf_counter()-t0)
            else:
                first_packet = False
                
            
            with self.signal_cond:
                t = time.perf_counter()
                self.add_new(packet)
                exec_time += (time.perf_counter()-t)
                self.signal_cond.notify_all()
            
            
            count +=1
            if not first_packet: loop_time += (time.perf_counter()-t0)
        
        
        self._print(f"SP Add execution time: {exec_time/count}")
        # self._print(f"SP Send count: {count}")
        # self._print(f"SP Frame wait: {frame_wait_time/count}")
        self._print(f"SP Packet wait: {packet_wait_time/(count-1)}")
        self._print(f"SP Total time: {loop_time/(count-1)}")
        self._running = False
        self.thread.join()
    
    def add_new(self, packet):
        
        if self.tmp_last_time > packet.timestamp:
            return
            
        # print(f"idx[{self.sig_idx}] : {packet.timestamp-self.tmp_last_time}")
        self.signal[packet.labels, self.sig_idx] = self.signal[packet.labels, self.sig_idx + self.sig_len] = packet.signal
        self.presence[:, self.sig_idx] = self.presence[:, self.sig_idx + self.sig_len] = False
        self.presence[packet.labels, self.sig_idx] = self.presence[packet.labels, self.sig_idx + self.sig_len] = True
        self.tmp_last_time = self.in_times[self.sig_idx] = self.in_times[self.sig_idx + self.sig_len] = packet.timestamp
        
        pos_ready = packet.labels & (self.running_counts>=POS_FRAMES)
        if np.any(pos_ready):
            # sig = self.signal[pos_ready, self.sig_idx+self.sig_len-POS_FRAMES+1:self.sig_idx+self.sig_len+1,:3]
            sigo = self.signal[pos_ready, self.sig_idx+self.sig_len-POS_FRAMES+1:self.sig_idx+self.sig_len+1]
            # timeso = self.signal[pos_ready, self.sig_idx+self.sig_len-POS_FRAMES+1:self.sig_idx+self.sig_len+1]
            # sig[...,3:] = sig[...,3:]-np.mean(sig[...,3:],axis=-2)
            # noise = self.signal[pos_ready, self.sig_idx+self.sig_len-POS_FRAMES+1:self.sig_idx+self.sig_len+1,3:]
            # noise = noise-np.mean(noise,axis=-2)#(np.linalg.pinv(sigt.T@sigt)@sigt.T)
            # sig[
            # for i in range(np.sum(pos_ready)):
                # sigt = sig[i]
                # sigt = (np.linalg.pinv(sigt@sigt.T)@sigt)
                # sig[i,:,:3] = sigt[:,:3]
                # print(sigt.shape)
            sig = sigo[...,:3]
            M = 1.0 / (np.mean(sig, axis=1,keepdims=True)+1e-9)
            sig = np.multiply(M, sig) #-1

            sig = np.einsum('ijk,lk->ijl',sig,self.P)
            alpha = np.std(sig[...,0], axis=1, keepdims=True) / (1e-9 + np.std(sig[...,1], axis=1, keepdims=True))
            # 300 x 300, 12
            Hn = np.add(sig[...,0], alpha * sig[...,1])
            sig = Hn - np.mean(Hn, axis=1, keepdims=True)
            
            # self.temp_ys[0] = sig[0]/np.std(sig[0])
            if INVERSE_TYPE in ['before','both']:
                for i in range(np.sum(pos_ready)):
                    # print(sigo.shape)
                    sigs = sigo[i,:,2:].copy()
                    sigs = (sigs-np.mean(sigs,axis=0,keepdims=True))#/np.std(sigs,axis=0,keepdims=True)
                    sigs[:,2] = sig
                    Q = sigs.T
                    sigt = (np.linalg.pinv(Q@Q.T)@Q).T
                    sig[i] = sigt[...,0]
                    # print(sigt.shape)
            # self.temp_ys[1] = sig[0]/np.std(sig[0])
            
            if STACKED_NOISE:
                noise = sigo[...,3:] 
                ##    0        1        2
                ## [labels, frames, channels]
                noise = (noise-np.mean(noise,axis=1,keepdims=True))#/np.std(noise,axis=1,keepdims=True)#/np.max(np.std(sig,axis=1,keepdims=True),axis=-1, keepdims=True)#* np.std(sig,axis=1,keepdims=True)[...,None]#/np.std(noise,axis=1,keepdims=True) * np.std(sig,axis=1,keepdims=True)[...,None]
                noise = np.squeeze(noise)
            sig = np.squeeze(sig)
            
            # self.temp_ys[0] = sig
            # for z in range(1,10):
                # self.temp_ys[z] = sigo[...,z]
            
            self.POS_sig[pos_ready, self.POS_circ_idxs] += sig
            self.POS_sig[pos_ready, self.POS_circ_idxs+self.POS_len] += sig
            
            cleaner = (self.POS_circ_idxs[-1]+1)%self.POS_len
            self.POS_sig[pos_ready, cleaner] = 0
            self.POS_sig[pos_ready, cleaner+self.POS_len] = 0
            
            if STACKED_NOISE:
                self.stacked_noise[pos_ready, self.POS_circ_idxs] += noise
                self.stacked_noise[pos_ready, self.POS_circ_idxs+self.POS_len] += noise
            
                self.stacked_noise[pos_ready, cleaner] = 0
                self.stacked_noise[pos_ready, cleaner+self.POS_len] = 0
            
        
        try:
            self.running_counts[~packet.labels] = 0
            self.running_counts[packet.labels] += 1
            # self.running_counts[(self.missing_counts<self.POS_FRAMES) & ~packet.labels] -= 1
        except Exception as e:
            pass
        
        self.HR_ready = (self.running_counts>=360) & ((self.running_counts-360) % STRIDE == 0)
        
        self.sig_idx +=1
        if self.sig_idx>=self.sig_len:
            self.sig_idx %= self.sig_len
            
        self.POS_circ_idxs +=1
        self.POS_circ_idxs %= self.POS_len
        
# class _OldSignalProcessor():
    # @staticmethod
    # def process_start_point(debug_lock, SP_FPtoSP, SP_SPtoIR, run_event, **kwargs):
        # instance = _SignalProcessor(debug_lock, SP_FPtoSP, SP_SPtoIR, run_event, **kwargs)
    
    # def _print(self, string):
        # self.debug_lock.acquire()
        # try:
            # print("{0: <16}: ".format(self.proc_name) + str(string), flush=True)
        # finally:
            # self.debug_lock.release()
    
    # P = np.array([[0, 1, -1], [-2, 1, 1]])
    # EPS = 1e-9
    
    # def __init__(self, debug_lock, SP_FPtoSP, SP_SPtoIR, run_event, **kwargs):
        # self.proc_name = current_process().name
        # if  self.proc_name != 'SignalProcessor':
            # raise RuntimeError("SignalProcessor must be instantiated on a seperate process, please use 'create_frame_processor' instead")
        
        
        # self.debug_lock = debug_lock
        # self.SP_FPtoSP = SP_FPtoSP
        # self.SP_SPtoIR = SP_SPtoIR
        # self.run_event = run_event
        
        
        # self.RGB_len            = O2_FRAMES
        # self.RGB_circ_len       = self.RGB_len*2
        # self.RGB_sig            = np.zeros((MAX_ENCODINGS, self.RGB_circ_len, 3))
        # self.RGB_present        = np.zeros((MAX_ENCODINGS, self.RGB_circ_len), dtype=bool)
        # self.RGB_circ_idx       = 0
        # # self.RGB_ready          = 
        
        # self.STD_sig            = np.zeros((MAX_ENCODINGS, self.RGB_circ_len, 3))
        
        # self.in_times           = np.zeros(self.RGB_circ_len, dtype=np.int64)
        # self.movement           = np.zeros((MAX_ENCODINGS,self.RGB_circ_len,2))
        
        # self.running_counts     = np.zeros(MAX_ENCODINGS, np.uint64)
        # self.missing_counts     = np.zeros(MAX_ENCODINGS, np.uint8)
        
        # self.POS_len            = np.maximum(HR_FRAMES, BR_FRAMES)
        # self.POS_circ_len       = self.POS_len*2
        # self.POS_sig            = np.zeros((MAX_ENCODINGS,self.POS_circ_len))
        # self.POS_std_sig        = np.zeros((MAX_ENCODINGS,self.POS_circ_len))
        # self.test_sig           = np.zeros(self.POS_circ_len)
        # self.POS_circ_idxs      = np.arange(POS_FRAMES, dtype=np.int32)
        
        # self.HR_ready           = np.zeros(MAX_ENCODINGS, dtype=bool)
        # self.BR_ready           = np.zeros(MAX_ENCODINGS, dtype=bool)
        # self.O2_ready           = np.zeros(MAX_ENCODINGS, dtype=bool)
        # self.HR_stride_counter  = np.zeros(MAX_ENCODINGS, np.uint8)
        # self.BR_stride_counter  = np.zeros(MAX_ENCODINGS, np.uint8)
        # self.O2_stride_counter  = np.zeros(MAX_ENCODINGS, np.uint8)
        
        # self.pre_filt           = butter(5, 3, fs=30, output='sos', btype='lowpass')
        # self.HR_filt            = butter(10, [0.65, 3], fs=30, output='sos', btype='bandpass')
        # self.BR_filt            = butter(10, [0.1, 0.6], fs=30, output='sos', btype='bandpass')
        
        # # self.RGB_cond         = Condition()
        # # self.pos_sig            = np.zeros((MAX_ENCODINGS,np.maximum(HR_FRAMES,BR_FRAMES)*2))
        # # self.start_counts       = np.zeros(MAX_ENCODINGS, dtype=np.uint8)
        
        # self.prev_label_mask    = np.zeros(MAX_ENCODINGS, dtype=bool)
        
        # self.thread = Thread(target=self._process_loop, name="SignalProcessor_proc_thread", daemon=True)
        
        
        # self.null_sent = False
        # self.prev_packet = FPtoSP_Packet(MAX_ENCODINGS)
        
        # self.run_event.wait()
        
        # self.debug_flag = True
        # self._input_loop()
        
    
    # def _process_loop(self):
        # l = []
        # def to_spect(x):
            # x = x * np.hanning(x.shape[-1])
            # fft = np.fft.rfft(x, n=8 * x.shape[-1])
            # spectrum = np.abs(fft)
            # freqs = np.linspace(0, 15 * 60, spectrum.shape[-1])
            # idx = np.where((freqs >= 0.65*60) & (freqs <= 3*60))
            # freqs = freqs[idx]
            # spectrum = spectrum[idx]
            # return freqs, spectrum
        
        # def animate(i):
            # if np.sum(self.running_counts>=POS_FRAMES)==0:
                # return line1,line2,line3,
            # y = sosfiltfilt(self.HR_filt, self.POS_sig[0,(self.POS_circ_idxs[-1]+self.POS_len-300 + 1):(self.POS_circ_idxs[-1]+self.POS_len+1)])
            # # y = self.POS_sig[0,(self.POS_circ_idxs[-1]+self.POS_len-300 + 1):(self.POS_circ_idxs[-1]+self.POS_len+1)]
            # # q = sosfiltfilt(self.HR_filt, self.POS_std_sig[0,(self.POS_circ_idxs[-1]+self.POS_len-300 + 1):(self.POS_circ_idxs[-1]+self.POS_len+1)])
            # # q = sosfiltfilt(self.HR_filt, self.test_sig[(self.POS_circ_idxs[-1]+self.POS_len-300 + 1):(self.POS_circ_idxs[-1]+self.POS_len+1)])
            # m = self.movement[0,((self.POS_circ_idxs[-1]%self.RGB_len)+self.RGB_len-300 + 1):((self.POS_circ_idxs[-1]%self.RGB_len)+self.RGB_len+1)]
            # # m = m/np.mean(m,axis=0)-1
            # m = sosfiltfilt(self.HR_filt, m,axis=0)
            # z = self.STD_sig[0,((self.POS_circ_idxs[-1]%self.RGB_len)+self.RGB_len-300 + 1):((self.POS_circ_idxs[-1]%self.RGB_len)+self.RGB_len+1)]
            # # z = z/np.mean(z,axis=0)-1
            # z = sosfiltfilt(self.HR_filt, z, axis=0)
            
            # q = np.concatenate([y[...,None],m,z],axis=-1)
            # pca = PCA(whiten=True)
            # a = pca.fit_transform(q)
            # # z = q*y
            # # mx = q[:,0]
            # # my = q[:,1]
            # # a = y
            # # nmx = mx/np.linalg.norm(mx)
            # # testx = np.dot(a,nmx) * nmx
            # # qn = q/np.linalg.norm(q)
            # # test = np.dot(a,qn) * qn
            # # q = a - test
            # # z = sosfiltfilt(self.HR_filt, q)
            # # q = z[...,0]
            # # z = z[...,1]
            # # z = sosfiltfilt(self.HR_filt, z, axis=0)
            # # m = m / np.mean(m,axis=0) -1
            # # data = y * np.hanning(len(y))
            # # fft = np.fft.rfft(data, n=8 * len(y))
            # # spectrum = np.abs(fft)
            # # freqs = np.linspace(0, 15 * 60, len(spectrum))
            # # idx = np.where((freqs >= 0.65*60) & (freqs <= 3*60))
            # # freqs = freqs[idx]
            # # spectrum = spectrum[idx]
            # # if spectrum.size==0:
                # # hr = 80
            # # else:
                # # hr = interp_peak(freqs, spectrum)
            # # # l.append(hr)
            # # line1.set_xdata(freqs)
            # # y = self.POS_sig[0,:self.POS_len]
            # # y = np.clip(y, -y.std()*3, y.std()*3)
            # # y = sosfiltfilt(self.HR_filt, y)
            # _, y = to_spect(q[...,0])
            # _, z = to_spect(q[...,1])
            # freq, q = to_spect(q[...,2])
            
            # line1.set_xdata(freq)
            # line2.set_xdata(freq)
            # line3.set_xdata(freq)
            
            # line1.set_ydata(y)
            # line2.set_ydata(z)
            # line3.set_ydata(q)
            # # line3.set_ydata(m[:,1])
            # # y = self.POS_sig[0,:self.POS_len]
            # # y = sosfiltfilt(self.HR_filt, y)
            # # line2.set_ydata(y)
            # ax.relim()
            # ax.autoscale_view()
            # return line1,line2,line3,
            
        # fig, ax = plt.subplots()
        
        # line1, = ax.plot([],[])
        # line2, = ax.plot([],[])
        # line3, = ax.plot([],[])
        # ani = FuncAnimation(fig, animate, interval=30, blit=True, save_count=50)
        # plt.show()
        
                
        
    # def _input_loop(self):
        # self.running = True
        # count = 0
        # packet_wait_time = 0
        # lock_time = 0
        # loop_time = 0
        # exec_time = 0
        
        # first_packet = True
        # self.thread.start()
        
        # while self.run_event.is_set():
            # t0 = time.perf_counter()
            # packet = self.SP_FPtoSP.recv()
            # if not first_packet:
                # packet_wait_time += (time.perf_counter()-t0)
            # else:
                # first_packet = False
            # t = time.perf_counter()
            # self.add_new(packet)
            # exec_time += (time.perf_counter()-t)
            # count +=1
            # if not first_packet: loop_time += (time.perf_counter()-t0)
        
        
        # self._print(f"SP Execution time: {exec_time/count}")
        # # self._print(f"SP Send count: {count}")
        # # self._print(f"SP Frame wait: {frame_wait_time/count}")
        # self._print(f"SP Packet wait: {packet_wait_time/(count-1)}")
        # self._print(f"SP Total time: {loop_time/(count-1)}")
        # self._running = False
        # self.thread.join()
    
    # def add_new(self, packet):
        
        # self.RGB_sig[packet.labels, self.RGB_circ_idx] = self.RGB_sig[packet.labels, self.RGB_circ_idx + self.RGB_len] = packet.colours
        # self.STD_sig[packet.labels, self.RGB_circ_idx] = self.STD_sig[packet.labels, self.RGB_circ_idx + self.RGB_len] = packet.stds
        # # print(packet.stds)
        # self.RGB_present[packet.labels, self.RGB_circ_idx] = self.RGB_present[packet.labels, self.RGB_circ_idx + self.RGB_len] = True
        # self.in_times[self.RGB_circ_idx] = self.in_times[self.RGB_circ_idx + self.RGB_len] = packet.timestamp
        # self.movement[packet.labels, self.RGB_circ_idx] = self.movement[packet.labels, self.RGB_circ_idx + self.RGB_len] = packet.movement
        # try:
            # self.running_counts[~packet.labels] = 0
            # # self.missing_counts[packet.labels] = 0
            # self.running_counts[(self.running_counts<self.RGB_len) & packet.labels] += 1
            # # self.running_counts[(self.missing_counts<self.POS_FRAMES) & ~packet.labels] -= 1
        # except Exception as e:
            # pass
            # # self._print(e)
            # # self._print((self.running_counts<self.RGB_len).dtype)
            # # self._print((self.running_counts<self.RGB_len).shape)
            # # self._print
        
        # pos_ready = packet.labels & (self.running_counts>=POS_FRAMES)
        # if np.any(pos_ready):
            # sig = self.RGB_sig[pos_ready, self.RGB_circ_idx+self.RGB_len-POS_FRAMES+1:self.RGB_circ_idx+self.RGB_len+1] 
            # M = 1.0 / (np.mean(sig, axis=1,keepdims=True)+1e-9)
            # sig = np.multiply(M, sig)

            # sig = np.einsum('ijk,lk->ijl',sig,self.P)
            # alpha = np.std(sig[...,0], axis=1, keepdims=True) / (1e-9 + np.std(sig[...,1], axis=1, keepdims=True))
            # Hn = np.add(sig[...,0], alpha * sig[...,1])
            # sig = Hn - np.mean(Hn, axis=1, keepdims=True)
            # sig = np.squeeze(sig) 
            
            # self.POS_sig[pos_ready, self.POS_circ_idxs] += sig
            # self.POS_sig[pos_ready, self.POS_circ_idxs+self.POS_len] += sig
            
            
            # # movement = self.movement[0, self.RGB_circ_idx+self.RGB_len-POS_FRAMES+1:self.RGB_circ_idx+self.RGB_len+1]
            # # stds = np.mean(self.STD_sig[0, self.RGB_circ_idx+self.RGB_len-POS_FRAMES+1:self.RGB_circ_idx+self.RGB_len+1],axis=-1,keepdims=True)
            # # a = np.concatenate([movement, stds], axis=-1)
            # # a = a/np.mean(a, axis=0, keepdims=True) - 1
            # # pca = PCA()
            # # b = pca.fit_transform(a)
            # # myb = myb - np.dot(myb,mxb) * mxb
            # # sig = self.RGB_sig[0, self.RGB_circ_idx+self.RGB_len-POS_FRAMES+1:self.RGB_circ_idx+self.RGB_len+1].copy()
            # # sig = sig - np.sum(b, axis=-1,keepdims = True)*np.mean(sig,keepdims=True)
            # # sig = sig[None,:]
            # # M = 1.0 / (np.mean(sig, axis=1,keepdims=True)+1e-9)
            # # sig = np.multiply(M, sig)

            # # sig = np.einsum('ijk,lk->ijl',sig,self.P)
            # # alpha = np.std(sig[...,0], axis=1, keepdims=True) / (1e-9 + np.std(sig[...,1], axis=1, keepdims=True))
            # # Hn = np.add(sig[...,0], alpha * sig[...,1])
            # # sig = Hn - np.mean(Hn, axis=1, keepdims=True)
            # # sig = np.squeeze(sig) 
            
            # # self.POS_std_sig[pos_ready, self.POS_circ_idxs] += (sig-np.sum(b,axis=-1))
            # # self.POS_std_sig[pos_ready, self.POS_circ_idxs+self.POS_len] += (sig-np.sum(b,axis=-1))
            
            
            # # a = sig
            # # b = std_sig
            # # nb = b/np.linalg.norm(b)
            # # test = np.dot(a,nb) * nb
            # # print(test)
            # # test = a - test
            # # print(test)
            # # self.test_sig[self.POS_circ_idxs] += test
            # # self.test_sig[self.POS_circ_idxs+self.POS_len] += test
            
            # # sig = self.RGB_sig[pos_ready, self.RGB_circ_idx+self.RGB_len-POS_FRAMES+1:self.RGB_circ_idx+self.RGB_len+1] - np.mean(self.movement[pos_ready, self.RGB_circ_idx+self.RGB_len-POS_FRAMES+1:self.RGB_circ_idx+self.RGB_len+1, None], axis=1, keepdims=True)
            # # M = 1.0 / (np.mean(sig, axis=1,keepdims=True)+1e-9)
            # # sig = np.multiply(M, sig)

            # # sig = np.einsum('ijk,lk->ijl',sig,self.P)
            # # alpha = np.std(sig[...,0], axis=1, keepdims=True) / (1e-9 + np.std(sig[...,1], axis=1, keepdims=True))
            # # Hn = np.add(sig[...,0], alpha * sig[...,1])
            # # sig = Hn - np.mean(Hn, axis=1, keepdims=True)
            
            # # sig = sosfiltfilt(self.pre_filt , sig, axis=-1) 
            # # sig = np.squeeze(sig) 
            
            # # self.POS_sig[pos_ready, self.POS_circ_idxs] += sig
            # # self.POS_sig[pos_ready, self.POS_circ_idxs+self.POS_len] += sig
            
            # cleaner = (self.POS_circ_idxs[-1]+1)%self.POS_len
            # self.POS_sig[pos_ready, cleaner] = 0
            # self.POS_sig[pos_ready, cleaner+self.POS_len] = 0
            
            
            # # self.POS_std_sig[pos_ready, cleaner] = 0
            # # self.POS_std_sig[pos_ready, cleaner+self.POS_len] = 0
            
            
            # # self.test_sig[cleaner] = 0
            # # self.test_sig[cleaner+self.POS_len] = 0
            
        # self.RGB_circ_idx +=1
        # if self.RGB_circ_idx>=self.RGB_len:
            # self.RGB_circ_idx %= self.RGB_len
            
        # self.POS_circ_idxs +=1
        # self.POS_circ_idxs %= self.POS_len