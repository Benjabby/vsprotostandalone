########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from multiprocessing import current_process, Event, Process, Pipe
from multiprocessing.shared_memory import SharedMemory
import time
import os
import json
# from itertools import filterfalse

import face_recognition
import psutil
import numpy as np
import mediapipe as mp
import cv2
# from skimage.transform import PiecewiseAffineTransform, warp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import geometric_slerp

from camera import SM_IMAGE_NAME, SM_TIME_NAME
from packets import FPtoIR_Packet, FPtoSP_Packet
from util import create_face_net, BoolIDList#, OneEuroFilter

TMP_SCALE       = 2
A_SCALE         = (TMP_SCALE-1)*0.5
B_SCALE         = A_SCALE+1

NORM_SKIN_SIZE  = 64
OUTSIDE_SCALE   = np.array([0.15,0.25])
BOX_ROI_SCALE   = 7
SKIN_IDX        = 10
SKIN_DIP        = 1/3

MIN_FACE_AREA   = 10000
MIN_SKIN_AREA   = 50*50

MAX_ENCODINGS   = 32
MAX_NUM_FACES   = 1

ENABLE_OEF      = True

def create_frame_processor(image_lock, debug_lock, new_frame_event, img_shape, **kwargs):
    
    if "FrameProcessor" in [p.name for p in psutil.process_iter()]:
        raise RuntimeError("FrameProcessor instance already running")
    
    run_event = Event()
    IR_FPtoIR, FP_FPtoIR = Pipe(True)
    SP_FPtoSP, FP_FPtoSP = Pipe(False)
    
    process = Process(target=_FrameProcessor.process_start_point, args=[image_lock, debug_lock, new_frame_event, FP_FPtoIR, FP_FPtoSP, img_shape, run_event],kwargs=kwargs, name="FrameProcessor", daemon=True)
    
    process.start()
    
    return process, run_event, IR_FPtoIR, SP_FPtoSP

class FaceManager:
    # The is one case missing that I havent included because its so unlikely to happen and I dont have time
    # That is when an occluded face and unoccluded face both become visible on the exact same frame with the ocluded face being first
    # in the ROI list but the second unnocluded face being a match for the encoding corresponding to the label that was just assigned to the occluded face.
    # Because the occluded face is included in 'new_labels' the unoccluded face wont check the encoding associated with the label that just been given
    # to the occluded face temporarily
    
    def __init__(self, debug_printer, w, h, encoding_update_frequencies=[3, 5, 15, 30, 60, 120, 240], filter_MC=0.5, filter_dMC=0.5, filter_beta=100, encoding_threshold=0.4, iou_threshold=0.4, rgb_area_threshold=100):
        # self.mask               = np.zeros(MAX_ENCODINGS, dtype=bool)
        self.face_encoder       = create_face_net()
        ## dry run to warm up 
        self.face_encoder.predict(np.zeros((1,160,160,3),dtype=np.float64),verbose=0)
        
        self.pix                = np.array([w,h])
        
        self.high               = np.array([255,255,255],dtype=np.uint8)
        self.low                = np.array([0,0,0],dtype=np.uint8)
        
        self.min_enc_area = (75*75)/(w*h)
        self.max_enc_area = (160*160)/(w*h)
        
        self.labels             = BoolIDList(MAX_ENCODINGS) # preserving the correct order of this is of utmost importance
        self.new_labels         = BoolIDList(MAX_ENCODINGS)
        self.pending_visibility = BoolIDList(MAX_ENCODINGS)
        
        self.unseen_labels      = np.ones(MAX_ENCODINGS, dtype=bool)
        self._occlusions        = np.zeros(MAX_ENCODINGS, dtype=bool)
        self.rgb_occlusions     = np.zeros(MAX_ENCODINGS, dtype=bool)
        self.encodings          = np.ones((MAX_ENCODINGS, 128), dtype=np.float64) # Need to use double precision float because otherwise geometric_slerp has problems
        self.enc_countdown      = np.zeros(MAX_ENCODINGS, dtype=np.int16)
        self.enc_freq_idx       = np.zeros(MAX_ENCODINGS, dtype=np.uint16)
        self.enc_discard_order  = np.zeros(MAX_ENCODINGS, dtype=np.uint8)
        self.prev_enc_area      = np.zeros(MAX_ENCODINGS, dtype=np.float64)
        self._rois              = np.zeros((MAX_ENCODINGS, 4, 2))
        
        self._prev_normed_skin  = np.zeros((MAX_ENCODINGS,NORM_SKIN_SIZE,NORM_SKIN_SIZE,3))
        self._prev_skin_size    = np.zeros((MAX_ENCODINGS))
        self._signal            = np.zeros((MAX_ENCODINGS, 12)) # R, G, B, std, dX, dY, dSize, absdiff mean, diff std, outside R, outside G, outside B
        
        self.MAX_ENCODINGS      = MAX_ENCODINGS
        self.encoding_threshold = encoding_threshold
        self.enc_freqs          = np.array(sorted(encoding_update_frequencies))
        self.iou_threshold      = iou_threshold
        self._swaps             = []
        self.n_faces            = 0
        self.latest_time        = 0
        
        self._rgb_vis_count     = np.zeros(MAX_ENCODINGS, dtype=np.uint64)
        # self.grace_period       = 47
        
        ## One Euro Filter Variables
        self.filter_MC          = np.full(self._rois.shape, filter_MC) * self.pix/np.min(self.pix)       # Minimum Cutoff            CONSTANT
        self.filter_dMC         = np.full(self._rois.shape, filter_dMC) * self.pix/np.min(self.pix)      # Derivative Minimum Cutoff CONSTANT
        self.filter_beta        = np.full(self._rois.shape, filter_beta) * self.pix/np.min(self.pix)     # Beta parameter            CONSTANT
        self.filter_prev_t      = np.zeros(self._rois.shape, dtype=np.uint64)                            # Previous Times
        self.filter_drois       = np.zeros_like(self._rois)                                              # Previous derivatives
        
        self.filter_MC.flags.writeable = False
        self.filter_dMC.flags.writeable = False
        self.filter_beta.flags.writeable = False
        
        self._print = debug_printer
        
    @property
    def boxes_float(self):
        return self._rois[self.labels,:2]
        
    @property
    def boxes_pix(self):
        return (self._rois[self.labels,:2] * self.pix).astype(np.int32)
        
    @property
    def boxes_clamped_float(self):
        return np.clip(self.boxes_float,0,1) 
    
    @property
    def boxes_clamped_pix(self):
        return np.clip(self.boxes_pix, 0, self.pix)
        
    @property
    def clamped_all_pix(self):
        return np.clip(self._rois[self.labels] * self.pix, 0, self.pix).astype(np.int32)
        
    @property
    def occlusions(self):
        return self._occlusions[self.labels]
        
    @property
    def rgb_vis_mask(self):
        return self.labels.mask & ~self.rgb_occlusions
        
    @property
    def rgb_vis_count(self):
        return self._rgb_vis_count[self.labels]
        
    @property
    def known_missing_idx(self):
        return self.labels.inverse_where(~self.unseen_labels)
        
    @property
    def known_missing_central_idx(self):
        return self.labels.inverse_where(~(self.unseen_labels | np.any(np.logical_or(self._rois<0, self._rois>=1), axis=(1,2))))
        
    @property
    def enc_update_idx(self):
        return self.labels.where(~self._occlusions & (self.enc_countdown<=0))
        
    @property
    def enc_hold_mask(self):
        return self.labels & (self.enc_countdown<=0) & self._occlusions
    
    @property
    def rgbs(self):
        return self._rgbs[self.labels]
        
    @property
    def stds(self):
        return self._stds[self.labels]
        
    @property
    def signal(self):
        return self._signal[self.labels]
        
    @property
    def fp_to_ir_packet(self):
        assert self.n_faces>=0
        return FPtoIR_Packet(roi_count=self.n_faces, rois=self._rois[self.labels], roi_labels=self.labels, roi_occluded=self.occlusions, rgb_occluded=self.rgb_occlusions[self.labels], vis_count=self.rgb_vis_count)
        
    @property
    def fp_to_sp_packet(self):
        assert self.n_faces>=0
        return FPtoSP_Packet(MAX_ENCODINGS, signal=self.signal, labels=self.labels, swaps=self._swaps, timestamp=self.latest_time)
        
    def _add_new(self, new_roi, frame_time, encoding=None, enc_area=None, new_label=None):

        # Get next usable label, replacing an older encoding if neccessary
        if new_label is None:
            new_label = np.argmin(self.enc_discard_order)
        # self._print(self.enc_discard_order)
        assert new_label not in self.labels
            
        self.enc_discard_order -= self.enc_discard_order[new_label] # Shift the priorities down to stop overflow creeping eventually
        self.enc_discard_order[new_label] = np.max(self.enc_discard_order)+1
        # self._print(self.enc_discard_order)
        
        if encoding is None:
            self.pending_visibility.append(new_label)
            self._occlusions[new_label] = True
            self.enc_countdown[new_label] = 0
            self.enc_freq_idx[new_label] = 0
            self.prev_enc_area[new_label] = self.min_enc_area
            # self.unseen_labels[new_label] = True # Dont change it. it could hold the encoding of a new face that comes along, in that case we need to do a delicate swap.
            # Leave the discard order as it is.
        else:
            self._occlusions[new_label] = False
            self.encodings[new_label] = encoding
            self.prev_enc_area[new_label] = enc_area
            self.enc_countdown[new_label] = self.enc_freqs[0]
            self.enc_freq_idx[new_label] = 0
            self.unseen_labels[new_label] = False
        
        self._prev_normed_skin[new_label] = 0
        self._prev_skin_size[new_label] = 0
        self.filter_prev_t[new_label] = frame_time
        self.filter_drois[new_label] = 0
        self._rois[new_label] = new_roi
        self.n_faces += 1
        self.new_labels.append(new_label)
        # self._print(f"New face {new_label} added")
        # self._print(f"Labels: {self.labels}")
        
    def _register_new_to_unlabeled(self, new_roi, label, frame_time, new_encoding, enc_area):
        # A new face has entered visibility that we have previously seen. We need to reset everything and merge the encodings
        self.enc_discard_order[label] = np.max(self.enc_discard_order)+1
        self.enc_discard_order -= np.min(self.enc_discard_order) # it might have been the lowest discard value so we shift it shift the minimum to keep it. if it isnt the minimum should be zero anyway
        self._occlusions[label] = False
        self.enc_countdown[label] = self.enc_freqs[self.enc_freq_idx[label]]
        self.unseen_labels[label] = False
        
        self._prev_normed_skin[label] = 0
        self._prev_skin_size[label] = 0
        self.filter_prev_t[label] = frame_time
        self.filter_drois[label] = 0
        self._rois[label] = new_roi
        self.n_faces += 1
        self.new_labels.append(label)
        self._merge_encodings(label, new_encoding, enc_area)
        self.prev_enc_area[label] = enc_area
        
    def _swap_new_to_unseen(self, new_roi, label, frame_time, new_encoding, enc_area):
        """
            A new face has entered visibility that we have previously seen, but the label for that encoding is currently used by an occluded face
            that has not yet been encoded. So we need to assign the new face to this label and put the occluded face on the next available label. 
        """
        occ_label = np.argmin(self.enc_discard_order)
        self._swaps.append((label, occ_label))
        assert occ_label not in self.labels
        self.labels.replace_value(label, occ_label)
        self.enc_discard_order[occ_label] = self.enc_discard_order[label]
        self.enc_freq_idx[occ_label] = self.enc_freq_idx[label]
        self.enc_countdown[occ_label] = self.enc_countdown[label]
        self._prev_normed_skin[occ_label] = self._prev_normed_skin[label]
        self._prev_skin_size[occ_label] = self._prev_skin_size[label]
        self._rois[occ_label] = self._rois[label]
        self._occlusions[occ_label] = True
        self.filter_prev_t[occ_label] = self.filter_prev_t[label]
        self.filter_drois[occ_label] = self.filter_dMC[label]
        self.prev_enc_area[occ_label] = self.min_enc_area
        
        self._cleanup_label(label)
        self._register_new_to_unlabeled(new_roi, label, frame_time, new_encoding, enc_area)
        
    def _register_unseen_to_unlabeled(self, old_label, true_label, new_encoding, enc_area):
        '''
            Called when a new face that was occluded (so couldn't be checked against existing faces) has become unoccluded and matches with an existing face,
            so we need to move all that data to the proper label.
            This all occurs after updating the old roi which was partially occluded.
            
        '''
        # self._print(f"old label {old_label}")
        # self._print(f"true label {true_label}")
        self._swaps.append((old_label, true_label))
        assert old_label in self.labels
        assert true_label not in self.labels
        self.labels.replace_value(old_label, true_label) # Needs to be put in the correct order so everything still aligns with mediapipes output
        self._occlusions[true_label] = False
        self.enc_freq_idx[true_label] = 0
        self.enc_countdown[true_label] = self.enc_freqs[self.enc_freq_idx[true_label]]
        # self.unseen_labels[true_label] = False # Needs to be done outside of this.
        self.filter_prev_t[true_label] = self.filter_prev_t[old_label]
        self.filter_drois[true_label] = self.filter_drois[old_label]
        self._merge_encodings(true_label, new_encoding, enc_area)
        self.prev_enc_area[true_label] = enc_area
        self._rois[true_label] = self._rois[old_label]
        self.enc_discard_order[true_label] = np.max(self.enc_discard_order)+1
        self.enc_discard_order -= np.min(self.enc_discard_order)
        self._prev_normed_skin[true_label] = self._prev_normed_skin[old_label]
        self._prev_skin_size[true_label] = self._prev_skin_size[old_label]
        
        
        
    def _cleanup_label(self, label):
        '''
            Resets the labels values but keeps the face encoding and resets its cleanup. Does not decrease the number of faces
        '''
        # self._print(f"Cleanup of label {label}")
        self._occlusions[label] = True
        #  Keep these as last known locations / values incase the face resurfaces.
        # self.enc_countdown[label] = 0
        # self.filter_prev_t[label] = 0 
        # self.filter_drois[label] = 0
        # self._rois[label] = 0 
        # self.prev_enc_area[label] = self.min_enc_area
        # Put this at the back of the discard - it will slowly decrease over time as this remains unregistered 
        self.enc_discard_order[label] = np.max(self.enc_discard_order)+1
        self.enc_discard_order -= np.min(self.enc_discard_order)
        if label in self.labels:
            self.labels.remove(label)
            self.n_faces -=1
        
    def _new_occlusion_test(self, new):
        return np.any(new<0) or np.any(new>=1) or ((new[1,0]-new[0,0])*(new[1,1]-new[0,1])*self.pix[0]*self.pix[1] < MIN_FACE_AREA)
        
    def _merge_encodings(self, label, new_encoding, new_area):
        # self._print(f"Previous area {self.prev_enc_area[label]}")
        # self._print(f"Current area {new_area}")
        weight = new_area/self.prev_enc_area[label] if new_area<self.prev_enc_area[label] else np.sqrt(new_area/self.prev_enc_area[label])
        weight = np.clip(weight/2, 0.1, 0.9)
        # self._print(f"Merged encodings with weight of {weight}")
        self.encodings[label] = geometric_slerp(self.encodings[label], new_encoding, weight)
        
    def _get_encoding(self, roi, in_frame):
        # ROI assumed to be unoccluded.
        # This function must be used as sparingly as possible!
        box = np.clip((roi[:2] * self.pix).astype(np.int32), 0, self.pix)
        img = in_frame[box[0,1]:box[1,1],box[0,0]:box[1,0]]
        img = (img-img.mean())/img.std()
        img = cv2.resize(img,(160,160),interpolation=cv2.INTER_AREA)
        enc = self.face_encoder.predict(np.expand_dims(img, axis=0),verbose=0)[0].astype(np.float64) 
        area = np.clip((roi[1]-roi[0]).prod(-1),self.min_enc_area,self.max_enc_area)
        return enc/np.linalg.norm(enc), area# We only compare by cosine similarity so might as well just store the normalized vectors to save having to compute the norm each comparison
    
    def _IOU(self,check_roi, rois):
        rois = rois.reshape(-1, 8)
        check_roi = check_roi.flatten()
        I = np.maximum(0,np.minimum(rois[:,2:4],check_roi[2:4])-np.maximum(rois[:,:2],check_roi[:2])).prod(-1)
        # self._print(I)
        return I / ((rois[:,2:4]-rois[:,:2]).prod(-1)+(check_roi[2:4]-check_roi[:2]).prod(-1)-I)
        
    def _find_removed_face(self, new_rois_og):
        """
        Called when a mediapipe outputs less faces than the previous frame. We need to work out which face was lost.
        
        Currently just keeping the simplest check for now.
        
        Process is as follows:
            First do simplest IOU check:
                Go through old ROIs, find intersection with new ROIs
                
        """
        
        remove_count = self.n_faces-len(new_rois_og)
        # self._print(f"Removing {remove_count} faces")
        old_rois = self._rois[self.labels].reshape(-1, 8)
        new_rois = new_rois_og.reshape(-1, 8)
        offset = 0
        # idx = list(range(self.n_faces))
        i=0
        found = []
        while i<len(new_rois):
            # self._print(f"Overlap area is: {np.maximum(0,np.minimum(old_rois[i+offset,2:4],new_rois[i,2:4])-np.maximum(old_rois[i+offset,:2],new_rois[i,:2])).prod(-1)}")
            if np.maximum(0,np.minimum(old_rois[i+offset,2:4],new_rois[i,2:4])-np.maximum(old_rois[i+offset,:2],new_rois[i,:2])).prod(-1)<=0:
                remove_count -=1
                found.append(self.labels[i+offset])
                offset += 1
                i -= 1
                if remove_count<= 0:
                    break
            i+=1
            
        if remove_count>0:
            return found + self.labels[-remove_count:] # Must* be the last ordered labels
        else:
            return found
        
        # new_roi_floats = np.array(new_roi_floats)[None]
        # old_roi_floats = np.array(old_roi_floats)[:,None]
        # new_roi_floats[...,2:] +=1
        # old_roi_floats[...,2:] +=1
        
        # intersect = (np.maximum(0,np.minimum(old_roi_floats[...,2:],new_roi_floats[...,2:])-np.maximum(old_roi_floats[...,:2],B[...,:2]))).prod(-1)>0
        # matches = np.any(intersect,axis=0)
        # if sum(matches)==removals and np.all(np.any(intersect,axis=1)):
            # idx=np.arange(len(old_roi_floats))
            # idx[~matches[1]].tolist()
        
    def _OEF(self, update_rois, frame_time):
        rois               = self._rois[self.labels]
        filter_MC          = self.filter_MC[self.labels]
        filter_dMC         = self.filter_dMC[self.labels]
        filter_beta        = self.filter_beta[self.labels]
        filter_prev_t      = self.filter_prev_t[self.labels]
        filter_drois       = self.filter_drois[self.labels]
        
        rois[filter_prev_t==0] = update_rois[filter_prev_t==0]
        filter_prev_t[filter_prev_t==0] = frame_time-1
        t_e = (frame_time - filter_prev_t)*1e-9
        dx = (update_rois - rois) / t_e
        if ENABLE_OEF:
            
            
            r = 2 * np.pi * filter_dMC * t_e
            a_d = r / (r + 1)
            dx_hat = a_d * dx + (1 - a_d) * filter_drois
            
            cutoff = filter_MC + filter_beta * np.abs(dx_hat)
            r = 2 * np.pi * filter_MC * t_e
            a = r / (r + 1)
            x_hat = a * update_rois + (1 - a) * rois
            
            self._rois[self.labels] = x_hat
            self.filter_drois[self.labels] = dx_hat
            self.filter_prev_t[self.labels] = frame_time
        else:
            self._rois[self.labels] = update_rois
            self.filter_drois[self.labels] = dx
            self.filter_prev_t[self.labels] = frame_time
        
    def unlink_all_faces(self):
        # self._print("Unlinking all faces")
        for removal in self.labels:
            self._cleanup_label(removal)
        
        
    def update(self, new_rois, frame_time, in_frame):
        # First step. Check for changes. 3 possibilities, more faces, less faces, or same number of faces
        assert self.n_faces==len(self.labels)
        self._swaps.clear()
        
        self.latest_time = frame_time
        
        # self._print("UPDATE CALLED")
        # self._print(new_rois.shape)
        # self._print(len(new_rois))
        if len(new_rois)>self.n_faces:
            # If it's a new face there's three things we need to do for each face
            
            add_count = len(new_rois)-self.n_faces
            additions = new_rois[:add_count] # New faces from mediapipe are always first.
            
            # self._print(f"{add_count} new faces")
            
            known_missing = self.known_missing_idx
            encodings_to_check = self.encodings[known_missing]
            check_prev = len(known_missing)>0
            
            iou_to_check = self.known_missing_central_idx # Only the central ROIS should be checked otherwise unknown faces coming from the edge will get matched incorrectly
            prev_seen = self._rois[iou_to_check]
            
            # self._print(f"IOUs to check: {iou_to_check}")
            
            for new in additions:
                # First check if it's occluded. If it is we can't do anything yet until it unoccludes itself.
                if self._new_occlusion_test(new):
                    self._add_new(new, frame_time, None)
                    continue
                    
                # Not occluded so can calculate encoding
                enc,enc_area = self._get_encoding(new, in_frame)
                
                if check_prev:
                    
                    # Check if its a known face from the known but missing encodings
                    results = np.dot(enc[None,:],encodings_to_check.T)
                    # uniqueness_score = 
                    if np.any(results>=self.encoding_threshold):
                        matched_label = known_missing[np.argmax(results)]
                        # self._print("Matched through encoding")
                        if matched_label in self.pending_visibility:
                            self._swap_new_to_unseen(new, matched_label, frame_time, enc, enc_area)
                        else:
                            self._register_new_to_unlabeled(new, matched_label, frame_time, enc, enc_area)
                        continue
                    
                    # Check if it overlaps enough with the last known location of any previous rois
                    results = self._IOU(new, prev_seen)
                    # self._print(f"IOU Results{results}")
                    if np.any(results>=self.iou_threshold):
                        matched_label = iou_to_check[np.argmax(results)]
                        # self._print("Matched through IOU")
                        if matched_label in self.pending_visibility:
                            self._swap_new_to_unseen(new, matched_label, frame_time, enc, enc_area)
                        else:
                            self._register_new_to_unlabeled(new, matched_label, frame_time, enc, enc_area) 
                        
                # if check_encodings:
                    
                    
                    else:
                        self._add_new(new, frame_time, encoding=enc, enc_area=enc_area)
                else:
                    # self._print("No encodings to check")
                    # No unaccounted for encodings so simply add new face
                    self._add_new(new, frame_time,encoding=enc, enc_area=enc_area)
            
            
        elif len(new_rois)<self.n_faces:
            # Face removed. We need to figure out which one(s) it was.
            removals = self._find_removed_face(new_rois)
            # self._print(f"Removing faces labeled: {removals}")
            for removal in removals:
                # self._print(f"Cleaning up face {removal}")
                self._cleanup_label(removal)
                    
        updates = new_rois[len(self.new_labels):]
        
        # ONE EURO FILTER UPDATE
        if len(updates)>0:
            self._OEF(updates, frame_time)
        
        # CHECK OCCLUSSION AGAIN
        curr_rois = self._rois[self.labels]
        if len(curr_rois)>0:
            self._occlusions[self.labels] = np.logical_or(np.any(np.logical_or(curr_rois<0, curr_rois>=1), axis=(1,2)), (curr_rois[:,1,0]-curr_rois[:,0,0])*(curr_rois[:,1,1]-curr_rois[:,0,1])*self.pix[0]*self.pix[1] < MIN_FACE_AREA)
        
        # CHECK FOR NEW-BUT-PREVIOUSLY-OCCLUDED FACES TO SEE IF THEY ARE NOW UNOCCLUDED AND RUN ENCODER CHECKS
        if len(set(self.pending_visibility) & set(self.labels))>0:
            missing_known_encodings = self.known_missing_idx
            encodings_to_check = self.encodings[missing_known_encodings]
            check_encodings = len(missing_known_encodings)>0
            
            now_visible = []
            transfered = []
            for l in self.pending_visibility:
                if not self._occlusions[l]:
                    now_visible.append(l)
                    enc, enc_area = self._get_encoding(self._rois[l], in_frame)
                    if check_encodings:
                        # Check if its a known face from the known but missing encodings
                        results = np.dot(enc[None,:],encodings_to_check.T)
                        if np.any(results>=self.encoding_threshold):
                            matched_label = missing_known_encodings[np.argmax(results)]
                            self._register_unseen_to_unlabeled(l,matched_label,enc, enc_area)
                            transfered.append(True)
                        else:
                            # No match for encodings so simply add encoding to label
                            self.encodings[l] = enc
                            self.prev_enc_area[l] = enc_area
                            self.enc_countdown[l] = self.enc_freqs[0]
                            self.enc_freq_idx[l] = 0 
                            transfered.append(False)
                    else:
                        # No unaccounted for encodings so simply add encoding to label
                        self.encodings[l] = enc
                        self.enc_countdown[l] = self.enc_freqs[0]
                        self.enc_freq_idx[l] = 0
                        transfered.append(False)
                    
            for n,t in zip(now_visible, transfered): # Want to remove them afterwards to avoid two newly unoccluded rois from recognizing as the same face. This does give precedence to the first ones but whatever.
                self.pending_visibility.remove(n)
                self.unseen_labels[n] = t # if transferred, n is the old label and we need to reset it fully
        
        self.enc_countdown[self.labels] -= 1
        
        # UPDATE ENCODINGS IF NECESSARY
        
        for l in self.enc_update_idx:
            enc, enc_area = self._get_encoding(self._rois[l], in_frame)
            self._merge_encodings(l, enc, enc_area)
            self.enc_freq_idx[l] = np.minimum(self.enc_freq_idx[l]+1, len(self.enc_freqs)-1)
            self.enc_countdown[l] = self.enc_freqs[self.enc_freq_idx[l]]
            # self._print(f"Encoding updated, next update in {self.enc_countdown[l]} frames")
            self.prev_enc_area[l] = np.maximum(enc_area, self.prev_enc_area[l])
        
        self.enc_countdown[self.enc_hold_mask] = 0
        
        
        self.labels = self.new_labels + self.labels # Append new to beginning
        if len(self.new_labels)>0:
            # self._print(f"Labels: {self.labels}")
            self.new_labels.clear()
        
        ## Get the RGB for all non-rgb_occluded. Normal occlusion does not neccessarily mean RGB occlusion and vice-versa
        curr_rois = self._rois[self.labels]
        if len(curr_rois)>0:
            self.rgb_occlusions[self.labels] = np.any(np.logical_or(curr_rois[:,2:,:]<0, curr_rois[:,2:,:]>=1), axis=(1,2))
            pix_rois = self.clamped_all_pix
            
            # self.rgb_occlusions[self.labels] |= self._quadrilateral_area_check(pix_rois) 
            for i,l in enumerate(self.labels.where(~self.rgb_occlusions)):
                (x1,y1),(x2,y2) = pix_rois[i,2:]
                (fx1,fy1),(fx2,fy2) = self._rois[l,2:]
                skin_size = (fx2-fx1)*(fy2-fy1)*MIN_FACE_AREA#*self.pix.prod()
                skin = in_frame[y1:y2,x1:x2]
                normed_skin = cv2.resize(skin, (NORM_SKIN_SIZE,NORM_SKIN_SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                normed_skin = normed_skin-np.mean(normed_skin,axis=(0,1),keepdims=True)
                if skin_size*self.pix.prod() < MIN_SKIN_AREA:
                    self.rgb_occlusions[l] = True
                    self._occlusions[l] = True
                    self._prev_skin_size[l] = skin_size
                    self._prev_normed_skin[l] = normed_skin
                    continue
                
                cut = skin.reshape(-1,3)
                cut = cut[np.logical_and((cut<self.high).any(axis=1),(cut>self.low).any(axis=1))]
                ds = self._prev_normed_skin[l]-normed_skin
                box_size = pix_rois[i,1] - pix_rois[i,0]
                ox1, oy1 = (pix_rois[i,0] - box_size*OUTSIDE_SCALE).astype(np.int32)
                ox2, oy2 = (pix_rois[i,1] + box_size*OUTSIDE_SCALE).astype(np.int32)
                in_mask = np.ones((in_frame.shape[0],in_frame.shape[1]),dtype=np.uint8)
                in_mask[oy1:oy2,ox1:ox2] = 0
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
                
                self._signal[l,:3]  = np.mean(cut,axis=0)
                self._signal[l,3]   = np.std(cut)
                self._signal[l,4:6] = np.mean(self.filter_drois[l],axis=0)/(self._rois[l,3]-self._rois[l,2])#*self.pix
                self._signal[l,6]   = skin_size - self._prev_skin_size[l]
                self._signal[l,7]   = np.mean(np.abs(ds))
                self._signal[l,8]   = np.mean(np.std(ds, axis=(0,1)))
                self._signal[l,9:]  = cv2.mean(in_frame, in_mask)[:3]
                # self._print(skin_size - self._prev_skin_size[l])
                self._prev_skin_size[l] = skin_size
                self._prev_normed_skin[l] = normed_skin.copy()
            
        self._rgb_vis_count[self.rgb_vis_mask] +=1
        self._rgb_vis_count[~self.rgb_vis_mask] = 0
            
class _FrameProcessor():
    @staticmethod
    def process_start_point(image_lock, debug_lock, new_frame_event, FP_FPtoIR, FP_FPtoSP, img_shape, run_event, **kwargs):
        instance = _FrameProcessor(image_lock, debug_lock, new_frame_event, FP_FPtoIR, FP_FPtoSP, img_shape, run_event, **kwargs)
    
    def _print(self, string):
        self.debug_lock.acquire()
        try:
            print("{0: <16}: ".format(self.proc_name) + str(string), flush=True)
        finally:
            self.debug_lock.release()
    
    def __init__(self, image_lock, debug_lock, new_frame_event, FP_FPtoIR, FP_FPtoSP, img_shape, run_event, single_stage=True, **kwargs):
        self.proc_name = current_process().name
        if  self.proc_name != 'FrameProcessor':
            raise RuntimeError("FrameProcessor must be instantiated on a seperate process, please use 'create_frame_processor' instead")
        
        self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)
        self.mp_kp = mp.solutions.face_mesh.FaceMesh(max_num_faces=MAX_NUM_FACES if single_stage else 1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        
        # self._stage = self._single_stage if single_stage else self._two_stage
        
        self.new_frame_event = new_frame_event
        self.image_lock = image_lock
        self.debug_lock = debug_lock
        self.FP_FPtoIR = FP_FPtoIR
        self.FP_FPtoSP = FP_FPtoSP
        self.run_event = run_event
        
        self.img_shape = img_shape
        self.aspect = img_shape[1]/img_shape[0]
        
        self.sm_image = SharedMemory(name=SM_IMAGE_NAME)
        self.frame = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.sm_image.buf)
        
        self.sm_time = SharedMemory(name=SM_TIME_NAME)
        self.frame_time = np.ndarray((), dtype=np.uint64, buffer=self.sm_time.buf)
        
        self.w = self.img_shape[1]
        self.h = self.img_shape[0]
        
        
        self.face_manager = FaceManager(self._print, self.w, self.h)
        
        self.null_sent = False
        self.prev_to_ir = FPtoIR_Packet()
        
        
        self.run_event.wait()
        
        self._process_loop()
        
        

    def _process_loop(self):
        count = 0
        frame_wait_time = 0
        lock_time = 0
        loop_time = 0
        exec_time = 0
        
        to_ir = FPtoIR_Packet()
        self.prev_to_ir = to_ir
        if self.FP_FPtoIR.poll():
            self.FP_FPtoIR.send(to_ir)
            self.FP_FPtoIR.recv()
            self.null_sent = True
        
        while self.run_event.is_set():
            t0 = time.perf_counter()
            self.new_frame_event.wait()
            frame_wait_time += (time.perf_counter()-t0)
            
            t = time.perf_counter()
            self.image_lock.acquire()
            lock_time += (time.perf_counter()-t)
            try:
                in_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                frame_time = self.frame_time.copy()
            finally:
                self.image_lock.release()
                self.new_frame_event.clear()
            
            t = time.perf_counter()
            in_frame.flags.writeable = False
            
            faces = self.mp_kp.process(in_frame)
            
            if not faces or not faces.multi_face_landmarks:
                to_sp = FPtoSP_Packet(MAX_ENCODINGS)
                self.FP_FPtoSP.send(to_sp)
                if not self.null_sent:
                    to_ir = FPtoIR_Packet()
                    self.prev_to_ir = to_ir
                    self.face_manager.unlink_all_faces()
                    if self.FP_FPtoIR.poll():
                        self.FP_FPtoIR.send(to_ir)
                        if self.run_event.is_set(): self.FP_FPtoIR.recv()
                        self.null_sent = True
                # self.RGB_pipe.send(RGB_out)
                continue
            
            new_rois = np.zeros((len(faces.multi_face_landmarks), 4, 2))
            
            for i, landmarks in enumerate(faces.multi_face_landmarks):
                xs = [l.x for l in landmarks.landmark]
                ys = [l.y for l in landmarks.landmark]
                new_rois[i,0,0] = np.min(xs)
                new_rois[i,0,1] = np.min(ys)
                new_rois[i,1,0] = np.max(xs)
                new_rois[i,1,1] = np.max(ys)
                
                bs = np.min(new_rois[i,1]-new_rois[i,0])/BOX_ROI_SCALE
                
                # new_rois[i,2,0] = xs[10]-bs
                # new_rois[i,2,1] = ys[10]-bs*(1-1/3)
                # new_rois[i,3,0] = xs[10]+bs
                # new_rois[i,3,1] = ys[10]+bs*(1+1/3)
                    
                new_rois[i,2,0] = np.maximum(xs[SKIN_IDX]-bs, new_rois[i,0,0])
                new_rois[i,2,1] = (ys[SKIN_IDX]-(bs*self.aspect)*(1-SKIN_DIP))
                new_rois[i,3,0] = np.minimum(xs[SKIN_IDX]+bs, new_rois[i,1,0])
                new_rois[i,3,1] = ys[SKIN_IDX]+(bs*self.aspect)*(1+SKIN_DIP)
                
            self.face_manager.update(new_rois, frame_time, in_frame)
            
            to_ir = self.face_manager.fp_to_ir_packet
            to_sp = self.face_manager.fp_to_sp_packet
            
            exec_time += (time.perf_counter()-t)
            
            self.FP_FPtoSP.send(to_sp)
            
            if self.FP_FPtoIR.poll():
                self.FP_FPtoIR.send(to_ir)
                if self.run_event.is_set(): self.FP_FPtoIR.recv()
                self.null_sent = False
                
            self.prev_to_ir = to_ir
            
            
            # self.RGB_pipe.send(RGB_out)
            count +=1
            loop_time += (time.perf_counter()-t0)
        
        self._print(f"FP Send count: {count}")
        self._print(f"FP Execution time: {exec_time/count}")
        self._print(f"FP Frame wait: {frame_wait_time/count}")
        self._print(f"FP Lock wait: {lock_time/count}")
        self._print(f"FP Total time: {loop_time/count}")
        self.mp_face.close()
        self.mp_kp.close()
        self.sm_image.close()
    