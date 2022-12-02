########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from multiprocessing import current_process, Event, Process
from multiprocessing.shared_memory import SharedMemory
import time
import ctypes

import psutil
import numpy as np
import mediapipe as mp
import cv2
from scipy.signal import sosfiltfilt, butter

from camera import SM_IMAGE_NAME
from packets import FPtoIR_Packet, TestPacket
from frame_processor import MAX_ENCODINGS

def create_image_renderer(image_lock, debug_lock, new_frame_event, FP_in, SP_in, img_shape, **kwargs):
    
    if "ImageRenderer" in [p.name for p in psutil.process_iter()]:
        raise RuntimeError("ImageRenderer instance already running")
    
    run_event = Event()
    close_event = Event()
    process = Process(target=_ImageRenderer.process_start_point, args=[image_lock, debug_lock, new_frame_event, FP_in, SP_in, img_shape, run_event, close_event],kwargs=kwargs, name="ImageRenderer", daemon=True)
    
    process.start()
    
    return process, run_event, close_event

class _ImageRenderer():
    @staticmethod
    def process_start_point(image_lock, debug_lock, new_frame_event, FP_in, SP_in, img_shape, run_event, close_event, **kwargs):
        instance = _ImageRenderer(image_lock, debug_lock, new_frame_event, FP_in, SP_in, img_shape, run_event, close_event, **kwargs)
    
    def _print(self, string):
        self.debug_lock.acquire()
        try:
            print("{0: <16}: ".format(self.proc_name) + str(string), flush=True)
        finally:
            self.debug_lock.release()
    
    def __init__(self, image_lock, debug_lock, new_frame_event, FP_in, SP_in, img_shape, run_event, close_event, **kwargs):
        self.proc_name = current_process().name
        if  self.proc_name != 'ImageRenderer':
            raise RuntimeError("ImageRenderer must be instantiated on a seperate process, please use 'create_image_renderer' instead")
        
        self.new_frame_event = new_frame_event
        self.image_lock = image_lock
        self.debug_lock = debug_lock
        self.FP_in = FP_in
        self.SP_in = SP_in
        self.run_event = run_event
        self.close_event = close_event
        
        # self.colours = []
        # for c in range(32):
            # self.colours.append(np.squeeze(cv2.cvtColor(np.array([[[128, np.random.randint(0,256),np.random.randint(0,256)]]],dtype=np.uint8), cv2.COLOR_LAB2BGR)).tolist())
        
        self.img_shape = img_shape
        self.sm_image = SharedMemory(name=SM_IMAGE_NAME)
        self.frame = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.sm_image.buf)
        self.in_w = self.img_shape[1]
        self.in_h = self.img_shape[0]
        
        cv2.namedWindow("Vital Signs Demo", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Vital Signs Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        user32 = ctypes.windll.user32
        cv2.resizeWindow("Vital Signs Demo", user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
        cv2.moveWindow("Vital Signs Demo",0,0)
        _, _, self.win_w, self.win_h = cv2.getWindowImageRect("Vital Signs Demo")
        self.out_image = np.zeros((self.win_h,self.win_w,3),dtype=np.uint8)
        self.in_aspect = self.in_w/self.in_h
        if self.in_aspect>=1:
            self.out_h = self.win_h
            self.out_w = int(self.out_h * self.in_aspect)
            self.y_border = 0
            self.x_border = (self.win_w - self.out_w)//2
        else:
            self.out_w = self.win_w
            self.out_h = int(self.out_w / self.in_aspect)
            self.y_border = (self.win_h - self.out_h)//2
            self.x_border = 0
            
        self.out_pix = np.array([self.out_w, self.out_h])
        self.pix_off = np.array([self.x_border, self.y_border])
        self.win_pix = np.array([self.win_w,self.win_h])
        
        load_img = np.zeros((self.win_h,self.win_w,3),dtype=np.uint8)
        fscale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, int(self.win_h/6), 6)
        (txtw, txth), txtb = cv2.getTextSize("Loading...", cv2.FONT_HERSHEY_SIMPLEX, fscale, 6)
        txtb+=6
        
        load_img = cv2.putText(load_img, "Loading...", ((self.win_w-txtw)//2, (self.win_h+txth)//2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (255,255,255), 6)
        cv2.imshow("Vital Signs Demo", load_img)
        cv2.waitKey(1)
        
        self.str_half_widths = np.zeros(MAX_ENCODINGS,np.uint32)
        self.str_heights = np.zeros(MAX_ENCODINGS,np.uint32)
        for i in range(MAX_ENCODINGS):
            (sw,sh),_ = cv2.getTextSize(f"Person {i}",  cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
            self.str_half_widths[i] = sw//2
            self.str_heights[i] = sh
            
        self.prev_packet = FPtoIR_Packet()
        
        self.last_HRs = np.zeros(MAX_ENCODINGS, dtype=np.uint8)
        
        # self.temp = np.zeros((MAX_ENCODINGS,600))
        # self.temp_idx = 0
        # self.temp_filt = butter(10, [0.65, 3], fs=30, output='sos', btype='bandpass')
        
        self.first_poll = True
        self.first_face = False
        ##### TODO Change of camera without changing process.
        
        self.run_event.wait()
        self._process_loop()
        
    def _draw_packet(self, packet, draw_img):
        
        if packet.roi_count>0:
            rois = np.clip(packet.rois * self.out_pix, 0, self.win_pix).astype(np.int32)
                        
            for roi,label,partial,rgb_occluded,vis_count in zip(rois, packet.roi_labels, packet.roi_occluded, packet.rgb_occluded, packet.vis_count):
                cnt = np.mean(roi,axis=1,keepdims=True)
                (x1,y1),(x2,y2) = roi[:2]#((roi-cnt)*1.1+cnt).astype(np.int32)
                # self.temp[label,self.temp_idx] = self.temp[label,self.temp_idx+300] = skcol[1]
                if rgb_occluded:
                    col = (0,0,255)
                    draw_img = cv2.rectangle(draw_img, (x1,y1), (x2, y2), col, 4)
                    draw_img = cv2.line(draw_img, (x1,y1), (x2,y2), col, 3)
                    draw_img = cv2.line(draw_img, (x1,y2), (x2,y1), col, 3)
                    draw_img = cv2.putText(draw_img, f"Move Closer", (x1,y2+self.str_heights[label]+4), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                elif partial:
                    col = (0,255,255)
                    draw_img = cv2.rectangle(draw_img, (x1,y1), (x2, y2), col, 4)
                    draw_img = cv2.line(draw_img, (x1,y1), (x2,y2), col, 3)
                    draw_img = cv2.line(draw_img, (x1,y2), (x2,y1), col, 3)
                    draw_img = cv2.putText(draw_img, f"Move Closer", (x1,y2+self.str_heights[label]+4), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4)
                else:
                    # col = skcol.astype(np.uint8)[::-1].tolist() #self.colours[label]
                    if vis_count<450:
                        prep_val = vis_count/450
                        prep_col = (0, 255, 255-int(255*prep_val))
                        # prep_rad = int((0.1+prep_val*0.9)*((x2-x1)+(y2-y1))/4)
                        prep_ang = int(vis_count/450*360) # Well that works out nicely. #int(prep_val*360)
                        prep_side = int(((x2-x1)+(y2-y1))/4)
                        # draw_img = cv2.circle(draw_img, (int((x1+x2)/2),int(int((y1+y2)/2))), prep_rad, prep_col, 4)
                        draw_img = cv2.ellipse(draw_img, (int((x1+x2)/2),int(int((y1+y2)/2))), (prep_side,prep_side), 270, 0, prep_ang, prep_col, -1) 
                    else:
                        # sig = self.temp[label,self.temp_idx:self.temp_idx+300]
                        # sig = sig/np.mean(sig)
                        # sig = sosfiltfilt(self.temp_filt, sig)
                        # heart_val = 1+2*np.maximum(0,sig[-48])
                        # self._print(heart_val)
                        # draw_img = cv2.circle(draw_img, (int((x1+x2)/2),int(int((y1+y2)/2))), int(heart_val*((x2-x1)+(y2-y1))/8), (0,0,170),-1)
                        draw_img = cv2.rectangle(draw_img, (x1,y1), (x2, y2), (10,220,15), 4)
                        if self.last_HRs[label]>0:
                            cv2.putText(draw_img, str(self.last_HRs[label]), (x1+(x2-x1)//3,y1-(y2-y1)//20), cv2.FONT_HERSHEY_SIMPLEX, 2, (10,15,220), 4)
                        # (x1,y1),(x2,y2) = roi[2:]
                        # draw_img = cv2.rectangle(draw_img, (x1,y1), (x2, y2), (10,220,15), 4)
                
                
                    # draw_img = cv2.putText(draw_img, f"Person {label}", ((x1+x2)//2 - self.str_half_widths[label],y2+self.str_heights[label]+4), cv2.FONT_HERSHEY_SIMPLEX, 2, (10,220,15), 4)
        # else:
            # self._print("What is going on")
        # self.temp_idx +=1
        # if self.temp_idx>= 300:
            # self.temp_idx %= 300
        return draw_img
        
    def _process_loop(self):
        count = 0
        skip_count = 0
        recieve_count = 0
        
        frame_wait_time = 0
        lock_time = 0
        loop_time = 0
        
        self.FP_in.send(None)
        
        while self.run_event.is_set():
            t0 = time.perf_counter()
            self.new_frame_event.wait()
            frame_wait_time += (time.perf_counter()-t0)
            
            t = time.perf_counter()
            self.image_lock.acquire()
            lock_time += (time.perf_counter()-t)
            try:
                draw_img = cv2.resize(self.frame, (self.out_w, self.out_h), interpolation=cv2.INTER_CUBIC)
            finally:
                self.image_lock.release()
                self.new_frame_event.clear()
            
            count +=1
            if self.first_poll or self.FP_in.poll():
                packet = self.FP_in.recv()
                self.FP_in.send(None)
                if self.first_poll:
                    self.first_poll = False
                    recieve_count = 0
                    continue
            
                recieve_count +=1
            
            # for ID, pix in packet.face_dict.items():
                # if ID>=len(self.colours):
                    # self.colours.append(np.squeeze(cv2.cvtColor(np.array([[[128, np.random.randint(0,256),np.random.randint(0,256)]]],dtype=np.uint8), cv2.COLOR_LAB2BGR)).tolist())
                # x1,y1,x2,y2 = pix
                # draw_img = cv2.rectangle(draw_img, (x1,y1), (x2, y2), self.colours[ID], 2)
                # draw_img = cv2.line(draw_img, (x1,y1), (x2,y2), self.colours[ID], 2)
                # draw_img = cv2.line(draw_img, (x1,y2), (x2,y1), self.colours[ID], 2)
                # draw_img = cv2.putText(draw_img, str(ID), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 2, self.colours[ID], 2)
                
            
            # if packet.main_roi_pix is not None:
                # x1,y1,x2,y2 = packet.main_roi_pix
                # draw_img = cv2.rectangle(draw_img, (x1,y1), (x2, y2), (0,255,int(packet.pending_frames/300 * 255)), 3)
            
                if self.SP_in.poll():
                    SP_packet = self.SP_in.recv()
                    self.last_HRs[SP_packet.labels] = SP_packet.HR_vals
                
                draw_img = self._draw_packet(packet, draw_img)
                
            # if packet.roi_count!=self.prev_packet.roi_count:
                # if packet.roi_count>self.prev_packet.roi_count:
                    # self._print("New ROI")
                # else:
                    # self._print("Lost ROI")
                # self._print("roi len: {}, label len: {}, warmup len: {}, temp len: {}".format(*[len(x) for x in [packet.roi_list, packet.roi_labels, packet.roi_warmups, packet.temp]]))
            # draw_img = cv2.circle(draw_img, (self.in_w//2, self.in_h//2), self.in_h//3, (0,0,0), 12)
            # for col in packet.temp:
                # ab = np.squeeze(cv2.cvtColor(col[None,None,:].astype(np.uint8),cv2.COLOR_BGR2Lab))[1:]
                # col = col.tolist()
                # ab = (ab.astype(np.float32)-128)
                # ab = ab/np.linalg.norm(ab)
                # ab = ab*(self.in_h//3) + np.array([self.in_w//2,self.in_h//2])
                
                # draw_img = cv2.circle(draw_img, (int(ab[0]),int(ab[1])), 5, col, -1)
                self.prev_packet = packet
            else:
                draw_img = self._draw_packet(self.prev_packet, draw_img)
                # self._print(self.prev_packet)
                
            self.out_image[self.y_border:self.y_border+self.out_h,self.x_border:self.x_border+self.out_w] = draw_img
            cv2.imshow("Vital Signs Demo", self.out_image)
            # self.temp[:] = packet.temp
            # cv2.imshow("temp",self.temp)
            k = cv2.waitKey(1)
            if k==ord("q") or k==27:
                self.close_event.set()
            
            
            loop_time += (time.perf_counter()-t0)
        
        cv2.destroyAllWindows()
        self._print(f"IR Frame count: {count}")
        self._print(f"IR Recieve count: {recieve_count}")
        self._print(f"IR Frame wait: {frame_wait_time/count}")
        self._print(f"IR Lock wait: {lock_time/count}")
        self._print(f"IR Total time: {loop_time/count}")
        self.sm_image.close()
    