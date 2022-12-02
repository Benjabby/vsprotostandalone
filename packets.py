########################################
# Code by Benjamin Tilbury             #
#       KTP Associate with UWS/Kibble  #
########################################

from dataclasses import dataclass, field
from typing import List

import numpy as np

from util import BoolIDList

@dataclass
class TestPacket:
    face_dict: dict = field(default_factory=lambda: {})

@dataclass
class FPtoIR_Packet:
    roi_count: int = 0
    rois: np.ndarray = None
    roi_labels: List = field(default_factory=lambda: [])
    # roi_warmups: List = field(default_factory=lambda: [])
    roi_occluded: np.ndarray = None
    rgb_occluded: np.ndarray = None
    vis_count: np.ndarray = None
    
@dataclass
class FPtoSP_Packet:
    max_encodings: int
    labels: BoolIDList = field(default_factory=lambda: None)
    signal: np.ndarray = None
    swaps: List = field(default_factory=lambda: [])
    timestamp: np.uint64 = 0
    def __post_init__(self):
        if self.labels is None:
            self.labels = BoolIDList(self.max_encodings)
    
@dataclass
class SPtoIR_Packet:
    HR_vals: np.ndarray = None
    labels: np.ndarray = None
    
@dataclass
class OldFPtoIR_Packet:
    roi_count: int = 0
    rois: np.ndarray = None
    roi_labels: List = field(default_factory=lambda: [])
    # roi_warmups: List = field(default_factory=lambda: [])
    roi_occluded: np.ndarray = None
    rgb_occluded: np.ndarray = None
    vis_count: np.ndarray = None
    temp: np.ndarray = None
    
@dataclass
class OldFPtoSP_Packet:
    max_encodings: int
    labels: BoolIDList = field(default_factory=lambda: None)
    colours: np.ndarray = None
    movement: np.ndarray = None
    stds: np.ndarray = None
    stdev: np.ndarray = None
    swaps: List = field(default_factory=lambda: [])
    timestamp: np.uint64 = 0
    def __post_init__(self):
        if self.labels is None:
            self.labels = BoolIDList(self.max_encodings)
    
