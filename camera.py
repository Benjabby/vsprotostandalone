########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from multiprocessing import Process, Lock, current_process, Event, Barrier, Pipe
from multiprocessing.shared_memory import SharedMemory
import time
from threading import Condition, Thread
import subprocess
import time
import warnings
import os
import time

import psutil
import cv2
import numpy as np

from util import create_time_ref, get_timestamp

SM_IMAGE_NAME   = "SM_IMAGE"
SM_TIME_NAME    = "SM_TIME"

from pygrabber.dshow_graph import FilterGraph
graph = FilterGraph()
try:
    DEVICES = graph.get_input_devices()
except ValueError:
    DEVICES = []
del graph

def create_live_camera(debug_lock, n_consumers, **kwargs):
    
    if "LiveCamera" in [p.name for p in psutil.process_iter()]:
        raise RuntimeError("LiveCamera instance already running")
    
    run_event = Event()
    
    image_lock = Lock()
    consumer_events = [Event() for _ in range(n_consumers)]
    img_shape_in, img_shape_out = Pipe(False)
    # close_barrier = Barrier(n_consumers+1)
    process = Process(target=_LiveCamera.process_start_point, args=[image_lock, debug_lock, run_event, consumer_events, img_shape_out],kwargs=kwargs, name="LiveCamera", daemon=True)
    
    process.start()
    
    img_shape = img_shape_in.recv()
    img_shape_in.close()
    if img_shape is None:
        process.kill()
        process.terminate()
        process.close()
        return False, None, None, None, None, None
    else:
        return True, process, image_lock, run_event, consumer_events, img_shape

class _LiveCamera:

    @staticmethod
    def process_start_point(image_lock, debug_lock, run_event, consumer_events, img_shape_out, **kwargs):
        time_ref = create_time_ref()
        
        instance = _LiveCamera(image_lock, debug_lock, run_event, consumer_events, img_shape_out, time_ref, **kwargs)
    
    def _print(self, string):
        self.debug_lock.acquire()
        try:
            print("{0: <16}: ".format(self.proc_name) + str(string), flush=True)
        finally:
            self.debug_lock.release()
            
    ## TODO with proper GUI
    def _settings_preview_thread(self):
        # ret, frame = self.stream.read()
        # font_scale = min(frame.shape[0],frame.shape[1])/25
        while self.run_settings:
            ret,frame = self.stream.read()
            if ret:
                cv2.imshow("Adjust Settings and Close Dialog Window",frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
    
    def __init__(self, image_lock, debug_lock, run_event, consumer_events, img_shape_out, time_ref, source=0, target_fps=30, target_size=(1080,720), debug_print=False, skip_settings=False, **kwargs):
        
        self.proc_name = current_process().name
        if  self.proc_name != 'LiveCamera':
            raise RuntimeError("LiveCamera must be instantiated on seperate process, please use 'create_live_camera' instead")
        
        self.sourceID = DEVICES.index(source) if isinstance(source,str) else source
        self.debug_print = debug_print
        self.time_ref = time_ref
        self.debug_lock = debug_lock
        self.image_lock = image_lock
        self.run_event = run_event
        self.consumer_events = consumer_events
        
        if debug_print: self._print("Attempting to create capture {}".format(source))
        self.stream = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        
        good = self.stream.isOpened()
        
        if good:
            if debug_print: self._print("Capture {} created".format(source))
            
                    
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
            
            self.stream.set(cv2.CAP_PROP_FPS, target_fps)
            
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.stream.set(cv2.CAP_PROP_AUTO_WB, 0)
            
            if not self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','U','Y','2')):
                if not self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','U','Y','V')):
                    self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
            
            self.stream.set(cv2.CAP_PROP_FPS, target_fps)
            
            if debug_print: self._print("Target properties set for {}\n\tFPS: {}\n\tWidth: {}\n\tHeight {}".format(source,target_fps,target_size[0],target_size[1]))

            
            self.fps = self.stream.get(cv2.CAP_PROP_FPS)
            self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if debug_print: self._print("True properties recieved for {}\n\tFPS: {}\n\tWidth: {}\n\tHeight {}".format(source,self.fps,self.width,self.height))
            
            ret,frame = self.stream.read()
            timestamp  = get_timestamp(self.time_ref)
            
            if not ret:
                good = ret
            
            if good:
                if debug_print: self._print("Initial read performed".format(source))
                
                if not skip_settings:
                    # self.stream.set(cv2.CAP_PROP_SETTINGS,1)
                    self.run_settings = True
                    tmp_thread = Thread(target=self._settings_preview_thread, daemon=True)
                    tmp_thread.start()
                    subprocess.run(f'res/ffmpeg.exe -f dshow -show_video_device_dialog true -i video="{DEVICES[source]}"',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    self.run_settings = False
                    tmp_thread.join()
                
                self.image_sm = SharedMemory(create=True, name=SM_IMAGE_NAME, size=frame.nbytes)
                self.frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=self.image_sm.buf)
                self.frame[:] = frame[:]
                
                self.time_sm = SharedMemory(create=True, name=SM_TIME_NAME, size=8)
                self.timestamp = np.ndarray((), dtype=np.uint64, buffer=self.time_sm.buf)
                self.timestamp[...] = timestamp
                
                img_shape_out.send(frame.shape)
                
                self.run_event.wait()
                self._main_loop()
        
        if not good:
            img_shape_out.send(None)
        
    def get_prop(self, prop):
        return self.stream.get(prop)
        
    def _main_loop(self):
        
        if self.debug_print: self._print("Loop started\n".format(self.sourceID))
        frame_time = 0
        event_time = 0
        frame_count = 0
        lock_time = 0
        read_time = 0
        while self.run_event.is_set():
            t0 = time.perf_counter()
            ret, img = self.stream.read()
            read_time += (time.perf_counter()-t0)
            if ret:
                t = time.perf_counter()
                self.image_lock.acquire()
                lock_time += (time.perf_counter()-t)
                try:
                    self.frame[:] = img[:]
                    self.timestamp[...] = get_timestamp(self.time_ref)
                    frame_count+=1
                finally:
                    t = time.perf_counter()
                    for cevent in self.consumer_events:
                        cevent.set()
                    event_time += (time.perf_counter()-t)
                    self.image_lock.release()
                
                
                    
            frame_time += (time.perf_counter()-t0)
            
        self._print(f"Cam Frame Rate: {self.fps}")
        self._print(f"Actual Frame Rate: {frame_count/frame_time}")
        self._print(f"Frame time: {frame_time/frame_count}")
        self._print(f"Event time: {event_time/frame_count}")
        self._print(f"Lock time: {lock_time/frame_count}")
        self._print(f"Read time: {read_time/frame_count}")
        self.stream.release()
        self.image_sm.close()
        self.image_sm.unlink()
    
    
    def __del__(self):
        self.stream.release()
        self.image_sm.unlink()
        

class OldCamera:
    def __init__(self, source=0, target_fps=30, target_size=(640,480), verbose=False):
        self.sourceID = DEVICES.index(source) if isinstance(source,str) else source
        
        self.verbose = verbose
        self.started = False
        
        self.stream = cv2.VideoCapture(self.sourceID, cv2.CAP_DSHOW)
        
        if verbose: print("Cam {}: Capture created".format(source))
        
        # self.stream.set(cv2.CAP_PROP_SETTINGS,1)
        self.stream.set(cv2.CAP_PROP_FPS, target_fps)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, target_size[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, target_size[1])
        
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','U','Y','2'))
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','U','Y','V'))
        self.stream.set(cv2.CAP_PROP_FPS, target_fps)
        
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.stream.set(cv2.CAP_PROP_AUTO_WB, 0)
        
        # self.stream.set(cv2.CAP_PROP_SETTINGS,1)
        
        if verbose: print("Cam {}: Target properties set\n\tFPS: {}\n\tWidth: {}\n\tHeight {}".format(source,target_fps,target_size[0],target_size[1]))

        
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if verbose: print("LiveStream {}: True properties recieved\n\tFPS: {}\n\tWidth: {}\n\tHeight {}".format(source,self.fps,self.width,self.height))

        self.frame = np.zeros((self.height,self.width,3),dtype=np.uint8)
        self.stream.read(self.frame)
        self.timestamp  = time.time_ns()
        if verbose: print("LiveStream {}: Initial read performed".format(source))

        self.cond = Condition()

        self.open    = True
        self.has_new = True
        
        self.thread = Thread(target=self._main_loop, args=())
        self.thread.daemon = True
        

    def start(self):
        if not self.started:
            self.started = True
            self.thread.start()
        else:
            warnings.warn("LiveStream {}: Attempt to start already started thread".format(self.sourceID),category=RuntimeWarning,stacklevel=2)
        
    def get_prop(self, prop):
        return self.stream.get(prop)
        
    def _main_loop(self):
        
        if self.verbose: print("LiveStream {}: Thread started\n".format(self.sourceID),flush=True,end='')
        self.time_ref = create_time_ref()
        frame_count = 0
        frame_time = 0
        while self.open:
            t = time.perf_counter()
            ret, img = self.stream.read()
            frame_time += (time.perf_counter()-t)
            if ret:
                with self.cond:
                    self.frame = img
                    self.frame.flags.writeable = False
                    self.timestamp = get_timestamp(self.time_ref)
                    self.has_new = True
                    self.cond.notify_all()
            frame_count+=1
        print(f"Frame time {frame_time/frame_count}")
        self.stream.release()
        
    def _has_new(self):
        return self.has_new
        
    def get(self):
        if not self.started:
            warnings.warn("LiveStream {}: Stream has not been started, please call 'start()'".format(self.sourceID),category=RuntimeWarning,stacklevel=2)
            return None, None
           
        with self.cond:
            self.cond.wait_for(self._has_new)
            self.has_new = False
            return self.frame, self.timestamp
        
    def close(self):
        self.open = False
        if not self.started:
            self.stream.release()

    def __del__(self):
        self.close()