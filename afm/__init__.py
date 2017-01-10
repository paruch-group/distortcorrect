# -*- coding: utf-8 -*-
# created by Benedikt Ziegler
# date: 07 December 2012

import feature_detection
import readAFMimage
import tools
import viewimage
from newsavetxt import savetxthd

__version__ = '0.0.5'

change_log = \
"""
version 0.0.5:
    - renaming some classes, modules, and functions and cleaning up
    - working on function documentation (in progress...)
    - added afm2pdf and afms2pdf script (commandline tool)
    - windows: context menu for afm2pdf
    - the size of scalebar label is now autocorrected
    - location of scalebar can be set like a plot legend
    - mac os x: context menu for afm2pdf
    - loading 'ibw' force curve files gives now an 'NotImplementedError'
    - added multiprocessing functionality to afm2pdf, the "-f" file
      declaration accepts now multiple files
    - new class Roughness in tools which calculates different statistical
      properties of a given 2d array
version 0.0.4:
    - import of submodules into afm
version 0.0.3:
	- extended tools set (determine miscut angle, determine step width)
version 0.0.2:
    - added image flattening funcitons (line and step flattening)
version 0.0.1:
    - load Atomic Force Microscopy image files for
        - Asylum Research Igor Binary Files (.ibw),
        - Nanonis (.sxm),
        - Bruker (.000),
        - WSxM (.txt)
    - plot and save images
"""


readAFMimage = readAFMimage