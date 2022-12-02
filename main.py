########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from dataclasses import dataclass
from multiprocessing import Process, Pipe, Array, Lock, freeze_support
from multiprocessing.shared_memory import SharedMemory
from threading import Thread, Event
import subprocess
import time
import os
import sys
import win32.win32gui as win32gui
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import cv2
import mediapipe as mp
import numpy as np
# from scipy import ndimage
from PyQt5 import QtCore, QtGui, QtWidgets

from camera import create_live_camera, SM_IMAGE_NAME, DEVICES
from frame_processor import create_frame_processor
from renderer import create_image_renderer
from signal_processor import create_signal_processor

TMP_SCALE = 2
A_SCALE = (TMP_SCALE-1)*0.5
B_SCALE = A_SCALE+1


# def warp_image(image, corners):
    # target = np.array([[0,0],[256,0],[256,256],[0,256]],dtype=np.float32)
    # mat = cv2.getPerspectiveTransform(corners, target)
    # out = cv2.warpPerspective(image, mat, (256,256), cv2.INTER_CUBIC)
    # return out

DEBUG_CAM_NAME = "Microsoft® LifeCam Cinema(TM)"

class MainProcessManager:
    def _print(self, string):
        self.debug_print_lock.acquire()
        try:
            print("{0: <16}: ".format("MainProcessManager") + str(string), flush=True)
        finally:
            self.debug_print_lock.release()

    def __init__(self, debug_print=False, force_cam=None):
        self.debug_print_lock = Lock()
        self.current_source = -1
        self.image_lock = None
        self.cam_process = None
        self.FP_process = None
        self.IR_process = None
        self.SP_process = None
        
        self.IR_FPtoIR = None
        self.SP_FPtoSP = None
        self.IR_SPtoIR = None
        self.debug_print = debug_print
        
        if force_cam is None:
            camselect = CameraPreviewManager()
            self.cam, self.res = camselect.wait_for()
        else:
            self.cam, self.res = force_cam, (1280, 720)
        
        self.running = True
        self.thread = Thread(name='MainProcess_main', target=self._main_loop, daemon=True)
        self.thread.start()
        
        # self.settings_thread = Thread(name='MainProcessManager_camera_settings', target=self._settings_preview_thread, daemon=True)
        # self.thread.start()
        
    def _main_loop(self):
        print("Loading. Please wait...")
        self.change_cam_source(self.cam)
        
        self.close_event.wait()
        self.close_current()
        print("Exited")
    
            
    def close_current(self):
        if self.IR_process is not None: self.IR_run_event.clear() 
        if self.SP_process is not None: self.SP_run_event.clear()
        if self.FP_process is not None: self.FP_run_event.clear()
        if self.cam_process is not None: self.cam_run_event.clear()
            
        if self.IR_process is not None:
            self.IR_process.join()
            print("Closed IR")
        if self.SP_process is not None:
            self.SP_process.join()
            print("Closed SP")
        if self.FP_process is not None:
            self.FP_process.join()
            print("Closed FP")
        if self.cam_process is not None:
            self.cam_process.join()
            print("Closed Camera")
            
        del self.IR_FPtoIR, self.SP_FPtoSP, self.IR_SPtoIR
            
    def change_cam_source(self, source):
        if isinstance(source, str): source = DEVICES.index(source)
        
        if source != self.current_source:
            
            self.close_current()
            
            success, self.cam_process, self.image_lock, self.cam_run_event, (self.FP_consumer, self.IR_consumer), self.img_shape = create_live_camera(self.debug_print_lock, 2, source=source,target_size=self.res, debug_print=self.debug_print,skip_settings=True)
            
            if not success:
                if self.debug_print: self._print(f"Camera ID {source-1} could not be opened. Recreating existing camera")
                source = self.current_source
                self.current_source = -1
                self.change_cam_source(source)
                return 
            else:
                if self.debug_print: self._print(f"Camera successfully changed to ID {source-1}")
                self.current_source = source
                
                self.FP_process, self.FP_run_event, self.IR_FPtoIR, self.SP_FPtoSP = create_frame_processor(self.image_lock,self.debug_print_lock,self.FP_consumer,self.img_shape)
                
                self.SP_process, self.SP_run_event, self.IR_SPtoIR = create_signal_processor(self.debug_print_lock, self.SP_FPtoSP)
                
                self.IR_process, self.IR_run_event, self.close_event = create_image_renderer(self.image_lock, self.debug_print_lock, self.IR_consumer, self.IR_FPtoIR, self.IR_SPtoIR, self.img_shape)
                
                if self.debug_print: self._print(f"Sending start signal to Camera process")
                self.cam_run_event.set()
                
                if self.debug_print: self._print(f"Sending start signal to Signal Processor process")
                self.SP_run_event.set()
                
                if self.debug_print: self._print(f"Sending start signal to Image Renderer process")
                self.IR_run_event.set()
                
                if self.debug_print: self._print(f"Sending start signal to Frame Processor process")
                self.FP_run_event.set()
                
                
class CameraSelectDialog(QtWidgets.QDialog):
    def __init__(self, main_obj, devices_dict, parent=None):
        super(CameraSelectDialog, self).__init__(parent)
        self.main_obj = main_obj
        self.devices_dict = devices_dict
        self.setWindowFlag(QtCore.Qt.WindowContextHelpButtonHint,False)
        if devices_dict is None:
            if len(DEVICES)==0:
                label = QtWidgets.QLabel("No supported devices found!")
                label.setStyleSheet('color: red')
                label.setFont(QtGui.QFont('Arial', 24))
                label.setAlignment(QtCore.Qt.AlignCenter)
            else:
                label = QtWidgets.QLabel("Camera devices detected, but none support the correct resolutions and/or encoding!")
                label.setStyleSheet('color: red')
                label.setFont(QtGui.QFont('Arial', 24))
                label.setAlignment(QtCore.Qt.AlignCenter)
                
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(label)
            self.setWindowTitle("Error")
        else:
            okay_button = QtWidgets.QPushButton('OK')
            okay_button.clicked.connect(self.on_clicked)
            okay_button.setFont(QtGui.QFont('Arial', 16))
            self.resbox = QtWidgets.QComboBox()
            self.resbox.setFont(QtGui.QFont('Arial', 16))
            self.resbox.currentTextChanged.connect(self.change_res_selection)
            self.combobox = QtWidgets.QComboBox()
            self.combobox.setFont(QtGui.QFont('Arial', 16))
            self.combobox.currentTextChanged.connect(self.change_selection)
            label = QtWidgets.QLabel("Select a camera from the dropdown menu and change its settings using the settings window. Ensure that auto white balance is on! Ensure that auto focus and auto exposure are disabled, and that exposure is -6 or lower!")
            label.setFont(QtGui.QFont('Arial', 16))
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setWordWrap(True)
            for device in devices_dict.keys():
                self.combobox.addItem(device)
            
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(label)
            lay.addWidget(self.combobox)
            lay.addWidget(self.resbox)
            lay.addWidget(okay_button)
            self.setWindowTitle("Select Camera")
    
    def change_res_selection(self, s):
        if s:
            li = s.split(" ")
            self.main_obj.res = (int(li[0]), int(li[2]))
    
    def change_selection(self, s):
        self.val = str(s)
        self.resbox.clear()
        for size in self.devices_dict[self.val]:
            self.resbox.addItem(size)
        
        self.main_obj.change_preview_camera(self.val)
        
    
    @QtCore.pyqtSlot()
    def on_clicked(self):
        if self.main_obj.check_settings_closed():
            self.accept()
        else:
            self.main_obj.focus_dialog()

class CameraPreviewManager:
    def __init__(self):
        
        self.devices_dict = self._determine_suitable_devices()
        self.current_selection = None
        self.res = (640, 480)
        
        self._temp_cam = None
        self.chosen_camera = None
        self.preview_thread = None
        self.settings_sub = None
        self.done_event = Event()
        # self.settings_thread = None
        app = QtWidgets.QApplication([sys.argv[0]])
        ex = CameraSelectDialog(self, self.devices_dict)
        ex.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # time.sleep(1)
        # ex.activateWindow()
        # ex.raise_()
        if ex.exec_() == QtWidgets.QDialog.Accepted:
            self.chosen_camera = ex.val
        else:
            print("Exiting...")
            app.quit()
            if self.settings_sub is not None and self.settings_sub.poll() is None:
                self.settings_sub.kill()
            exit()
            
        app.quit()
        self.done_event.set()
        
    def _determine_suitable_devices(self):
        devices = {}
        try:
            for device in DEVICES:
                supported_resolutions = self._get_supported_resolutions(device)
                if supported_resolutions is not None:
                    devices[device] = supported_resolutions
        except: # Very bad practice
            return None
            
        if len(devices)==0:
            return None
        else:
            return devices
        
    def _get_supported_resolutions(self, device_name):
        # Only present resolutions that can run at 30
        out = subprocess.run(f'res/ffmpeg.exe -f dshow -list_options true -i video="{device_name}"',capture_output=True)
        sizes = set()
        string = out.stderr.decode() # FFMPEG uses stderr
        string = string[string.find("DirectShow video device options"):]
        spl    = string.splitlines()[2:-1]
        for sp in spl:
            if not ("pixel_format" in sp or "mjpeg" in sp):
                continue
            spmin = sp[sp.find("min"):]
            end = spmin.find(" ",spmin.find("fps="))
            if end == -1: end = len(spmin)
            fps = float(spmin[spmin.find("fps=")+4:end])
            if fps>30:
                continue
            sp = sp[sp.find("max"):]
            end = sp.find(" ",sp.find("fps="))
            if end == -1: end = len(sp)
            fps = float(sp[sp.find("fps=")+4:len(sp)+1+sp.find(" ",sp.find("fps="))])
            if fps<30:
                continue
            size = sp[sp.find("s=")+2:sp.find(" ",sp.find("s="))].replace("x", " x ")
            
            li = size.split(" ")
            if min(int(li[0]),int(li[2]))>=480:
                sizes.add(size)
                # continue
                
        sizes = sorted(list(sizes),key=lambda x: int(x.split()[0]), reverse=True)
        
        if len(sizes)==0:
            return None
        else:
            return sizes
        # self.resbox.clear()
        # for size in sizes:
            # self.resbox.addItem(size)
        
    def focus_dialog(self):
        try:
            hwnd = win32gui.FindWindowEx(0,0,0, "0001 Properties")
            win32gui.SetForegroundWindow(hwnd)
        except:
            time.sleep(1)
            try:
                hwnd = win32gui.FindWindowEx(0,0,0, "0001 Properties")
                win32gui.SetForegroundWindow(hwnd)
            except:
                pass
        
    def check_settings_closed(self):
        return self.settings_sub.poll() is not None
        
    def change_preview_camera(self, new):
        self.current_selection = new
        # print(new)
        if self.preview_thread is not None:
            self.run_settings = False
            self.preview_thread.join()
        if self.settings_sub is not None and self.settings_sub.poll() is None:
            self.settings_sub.kill()
        
        self._temp_cam = cv2.VideoCapture(DEVICES.index(new), cv2.CAP_DSHOW)
        
        self.settings_sub = subprocess.Popen(f'res/ffmpeg.exe -f dshow -show_video_device_dialog true -i video="{new}"',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        
        self.preview_thread = Thread(name='MainProcess_camera_preview', target=self._settings_preview_loop, daemon=True)
        self.run_settings = True
        self.preview_thread.start()
        
        time.sleep(1)
        self.focus_dialog()
        
    def _settings_preview_loop(self):
        # ret,frame = self._temp_cam.read()
        # if ret:
            # cv2.imshow("Adjust Settings and Close Dialog Window",frame)
        # cv2.waitKey(1)
        
        while self.run_settings:
            ret,frame = self._temp_cam.read()
            if ret:
                cv2.imshow("Adjust Settings and Close Dialog Window",frame)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        self._temp_cam.release()
        self._temp_cam = None
        
    def wait_for(self):
        self.done_event.wait()
        if self.preview_thread is not None:
            self.run_settings = False
            self.preview_thread.join()
        if self.settings_sub is not None and self.settings_sub.poll() is None:
            self.settings_sub.kill()
            
        return self.current_selection, self.res

if __name__=="__main__":
    freeze_support()
    if len(sys.argv)>1 and "skip" in sys.argv[1:]:
        M = MainProcessManager(force_cam=DEBUG_CAM_NAME)
    else:
        M = MainProcessManager()
    M.thread.join()
    
    
        

# KP_INDICES = [109,338,336,107]
# mp1 = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
# mp2 = mp.solutions.face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.7, min_tracking_confidence=0.5)
# c = OldCamera(source=1)
# w = c.width
# h = c.height
# c.start()

# prev_roi = None

# while True:
    
    # frame, frame_time = c.get()
    
    # in_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # in_frame.flags.writeable = False
    # faces = mp1.process(in_frame)
    
    # # if not faces or not faces.detections:
        # # # prev_roi = None
        # # continue
    # # breakpoint()
    
    # drawable = frame.copy()
    
    # # detections = detections
    
    # # big_id = max(range(len(faces.detections)), key=lambda i: faces.detections[i].location_data.relative_bounding_box.width*faces.detections[i].location_data.relative_bounding_box.height)
    
    # # detection = faces.detections.pop(big_id).location_data.relative_bounding_box
    
    
    # # # mp.solutions.drawing_utils.draw_detection(drawable, detection)
    
    # # mx1 = (detection.xmin - detection.width*A_SCALE)
    # # my1 = (detection.ymin - detection.height*A_SCALE)
    # # mx2 = (detection.xmin + detection.width*B_SCALE)
    # # my2 = (detection.ymin + detection.height*B_SCALE)
    
    # # x1 = max(int(mx1*w),0)
    # # y1 = max(int(my1*h),0)
    # # x2 = min(int(mx2*w),w)
    # # y2 = min(int(my2*h),h)
    
    # # if (x2-x1)<2 or (y2-y1)<2: continue
    
    # # if prev_roi is not None:
        # # px,py,pw,ph = prev_roi
        # # if not (px <= detection.xmin + detection.width and detection.xmin <= px + pw and py <= detection.ymin + detection.height and detection.ymin <= py + ph):
            # # print("FACE SWAP")
    
    # # prev_roi = [detection.xmin,detection.ymin,detection.width,detection.height]
    
    # res = mp2.process(in_frame)
    # # res = mp2.process(np.ascontiguousarray(in_frame[y1:y2,x1:x2]))
    
    # if not res or not res.multi_face_landmarks:
        # continue
    
    # # breakpoint()
    # # breakpoint()
    # # face_landmarks = res.multi_face_landmarks[0]
    # # breakpoint()
    
    
    # # v = np.array([[face_landmarks.landmark[i].x*(x2-x1),face_landmarks.landmark[i].y*(y2-y1)] for i in KP_INDICES],dtype=np.float32)
    
    # # warped = warp_image(drawable[y1:y2,x1:x2], v)
    
    # # cv2.imshow("warped", warped)
    # # L = cv2.cvtColor(warped,cv2.COLOR_BGR2LAB)[...,0]
    # # Y = cv2.Sobel(L,cv2.CV_16S,0,1) * 10
    # # X = cv2.Sobel(L,cv2.CV_16S,1,0) * 10
    
    # # cv2.imshow("warped Y", cv2.convertScaleAbs(Y))
    # # cv2.imshow("warped X", cv2.convertScaleAbs(X))
    # for face_landmarks in res.multi_face_landmarks:
        # mp.solutions.drawing_utils.draw_landmarks(
          # image=drawable,
          # landmark_list=face_landmarks,
          # connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
          # landmark_drawing_spec=None,
          # connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
    # # mp.solutions.drawing_utils.draw_landmarks(
      # # image=annotated_image,
      # # landmark_list=face_landmarks,
      # # connections=mp_face_mesh.FACEMESH_CONTOURS,
      # # landmark_drawing_spec=None,
      # # connection_drawing_spec=mp_drawing_styles
      # # .get_default_face_mesh_contours_style())
    
    # # drawable = cv2.rectangle(drawable, (x1,y1), (x2, y2), (255,255,255), 3)
    # # mp.solutions.drawing_utils.draw_detection(drawable, detection)
    # f = res.multi_face_landmarks[0].landmark
    
    # arr = np.array([[l.x,l.y,l.z] for l in f])
    # arr = arr[KP_INDICES,:]
    # # print(f"\r{str(arr[0,2]):<10},{str(arr[1,2]):<10},{str(arr[2,2]):<10},{str(arr[3,2]):<10}",end="")
    
    # # for detection in faces.detections:
        # # # breakpoint()
        
        # # mx1 = (detection.location_data.relative_bounding_box.xmin - detection.location_data.relative_bounding_box.width*0.1)
        # # my1 = (detection.location_data.relative_bounding_box.ymin - detection.location_data.relative_bounding_box.height*0.2)
        # # mx2 = (detection.location_data.relative_bounding_box.xmin + detection.location_data.relative_bounding_box.width*1.1)
        # # my2 = (detection.location_data.relative_bounding_box.ymin + detection.location_data.relative_bounding_box.height*1.2)
    
        # # x1 = max(int(mx1*w),0)
        # # y1 = max(int(my1*h),0)
        # # x2 = min(int(mx2*w),w)
        # # y2 = min(int(my2*h),h)
    
        # # drawable = cv2.rectangle(drawable, (x1,y1), (x2, y2), (0,0,255), 3)
        # # drawable = cv2.line(drawable, (x1,y1), (x2,y2), (0,0,255), 3)
        # # drawable = cv2.line(drawable, (x1,y2), (x2,y1), (0,0,255), 3)
            
    # cv2.imshow('f', drawable)
    # k = cv2.waitKey(1)
    # if k == ord("q"):
        # c.close()
        # c.thread.join()
        # break
    # elif k == ord("w"):
        # breakpoint()
