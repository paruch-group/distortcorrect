# -*- coding: utf-8 -*-
# created by Benedikt Ziegler
# date: 22 November 2012

import re


class AFMImage:
    """
    Declaration of dictionary with the standard values for afm images.

    This function should be used for a standardized output for ReadIBW,
    ReadSXM, ReadWSxM, etc.
    """

    __version__ = 2

    def __init__(self):
        self.pixels_x = None
        self.pixels_y = None
        self.scan_center_x = None
        self.scan_center_y = None
        self.scansize_x = None
        self.scansize_y = None
        self.pixsize_x = None
        self.pixsize_y = None
        self.scanangle = None
        self.notes = None
        self.channelnames = None
        self.channelunits = None
        self.data = None
        self.metadata = None
        self.filename = None
        self.filetype = None
        self.date = None
        self.time = None
        self.imagingmode = None

    def _calc_pixSize(self):
        self.pixsize_x = self.scansize_x / self.pixels_x
        self.pixsize_y = self.scansize_y / self.pixels_y


def convert_length2m(length, unit):
    if unit == 'nm':
        scale = 1e-9
    elif unit == 'um':
        scale = 1e-6
    elif unit == 'mm':
        scale = 1e-3
    elif unit == 'm':
        scale = 1
    else:
        raise ValueError('Unknown unit type!')
    return length * scale


def find_correct_unit(channelname):
    if 'phase' in channelname.lower():
        siunit = 'deg'
    elif 'frequency' in channelname.lower():
        siunit = 'Hz'
    elif 'height' in channelname.lower():
        siunit = 'm'
    elif 'amplitude' in channelname.lower():
        siunit = 'm'
    else:
        siunit = 'nd'
    return siunit


def ibwChannel2Unit(channelname):
    channelname = channelname.split('Retrace')[0].split('Trace')[0]
    if 'Nap' in channelname:
        channelname = channelname.split('Nap')[1]

    dataTypeClass = ''
    datatype = channelname

    if 'Defl' in channelname:
        if 'DeflV' in channelname:
            dataTypeClass = "Volts"
        dataType = 'Deflection'
    elif 'Meters' in channelname:
        datatype = 'ZPiezo'
    elif 'Volts' in channelname:
        datatype = 'Volts'
        dataTypeClass = 'Volts'
    elif 'Voltage' in channelname:
        datatype = 'Volts'
        dataTypeClass = 'Volts'
    elif 'Potential' in channelname:
        datatype = 'Volts'
        dataTypeClass = 'Volts'
    elif 'Bias' in channelname:
        datatype = 'Volts'
        dataTypeClass = 'Volts'
    elif 'ZSensor' in channelname:
        datatype = 'ZLVDT'
    elif 'ZSnsr' in channelname:
        datatype = 'ZLVDT'
    elif 'Raw' in channelname:
        if 'RawV' in channelname:
            dataTypeClass = 'Volts'
        datatype = 'ZLVDT'
    elif 'LVDT' in channelname:
        if len(channelname) > 4:
            datatype = channelname[0] + 'LVDT'
        else:
            datatype = 'ZLVDT'
    elif 'Sep' in channelname or 'Ind' in channelname:
        datatype = 'ZLVDT'
    elif 'Invols' in channelname:
        # keep ahead of Amp (AmpInvols)
        datatype = 'Invols'
    elif 'KSample' in channelname:
        datatype = 'K'
    elif 'Amp' in channelname:
        if 'AmpV' in channelname:
            dataTypeClass = 'Volts'
        elif 'DriveAmplitude' in channelname:
            dataTypeClass = 'Volts'
        datatype = 'Amplitude'
    elif 'Amp2' in channelname:
        if 'Amp2V' in channelname:
            dataTypeClass = 'Volts'
        elif 'DriveAmplitude' in channelname:
            dataTypeClass = 'Volts'
        datatype = 'Amplitude2'
    elif 'Phase' in channelname or 'Phas2' in channelname or \
                    'Phasd' in channelname:
        datatype = 'Phase'
    elif 'Current' in channelname or \
                    'Cu' in channelname or 'Cur' in channelname:
        datatype = 'Current'
    elif 'Count2' in channelname or 'c2' in channelname:
        datatype = 'Count2'
    elif 'Count1' in channelname or 'c1' in channelname:
        datatype = 'Count1'
    elif 'Count0' in channelname or 'c0' in channelname:
        datatype = 'Count0'
    elif 'Count' in channelname or 'Ct' in channelname:
        datatype = 'Count'
    elif 'Optical' in channelname or 'Op' in channelname:
        datatype = 'Count'
    elif 'Calc' in channelname:
        if channelname[-1] == 'B':
            datatype = 'UserCalcB'  # must be ahead of lat.
        else:
            datatype = 'UserCalc'  # must be ahead of lat.
    elif 'Cb' in channelname:
        datatype = 'UserCalcB'  # must be ahead of lat.
    elif 'Ca' in channelname:
        datatype = 'UserCalc'  # must be ahead of lat.
    elif 'Lat' in channelname:
        datatype = 'Lateral'
        userNumStr = ''
    elif 'User' in channelname:
        datatype = 'User'
        userNumStr = re.findall(r'\d+', channelname)[-1]
    elif 'In0' in channelname:
        datatype = 'User'
        userNumStr = '0'
    elif 'In1' in channelname:
        datatype = 'User'
        userNumStr = '1'
    elif 'In2' in channelname:
        datatype = 'User'
        userNumStr = '2'
    elif 'THtDr' in channelname or 'TipHeaterDrive' in channelname or \
                    'Td' in channelname:
        datatype = 'TipHeaterDrive'
    elif 'THtPw' in channelname or 'TipHeaterPower' in channelname or \
                    'Tp' in channelname:
        datatype = 'TipHeaterPower'
    elif 'Temp' in channelname or 'TipTemperature' in channelname or \
                    'Tk' in channelname:
        datatype = 'Temperature'
    elif 'Height' in channelname or 'Z' in channelname:
        datatype = 'ZPiezo'
    elif 'Indentation' in channelname:
        datatype = 'ZPiezo'
    elif 'Freq' in channelname or 'Rate' in channelname:
        # Keep ahead of force (ForceScanRate)
        # also keep ahead of Drive (DriveFrequency)
        datatype = 'Frequency'
    elif 'Fr' in channelname or 'Rate' in channelname:
        datatype = 'Frequency'
    elif 'Drive' in channelname:
        if 'DriveV' in channelname:
            dataTypeClass = 'Volts'
        datatype = 'ZPiezo'
    elif 'Force' in channelname or 'Adhesion' in channelname:
        # Keep ahead of dist
        if 'Dist' in channelname:
            dataTypeClass = 'Meters'
        else:
            dataTypeClass = 'Force'
        datatype = 'Deflection'
    elif 'WorkArea' in channelname:
        dataTypeClass = "SurfaceEnergy"
        datatype = "SurfaceEnergy"
    elif 'Work' in channelname:
        dataTypeClass = "Work"
        datatype = "Work"
    elif 'Plasticity' in channelname:
        dataTypeClass = "Factor"
        datatype = "Factor"
    elif 'Dist' in channelname:
        datatype = 'ZPiezo'
    elif 'Length' in channelname:
        datatype = 'ZPiezo'
    elif 'Piezo' in channelname:
        datatype = channelname[0] + 'Piezo'
    elif 'Sensor' in channelname:
        if 'XSensor' in channelname:
            datatype = 'X'
        else:
            datatype = 'Y'
        datatype += 'LVDT'
    elif 'Y' in channelname or 'X' in channelname:
        datatype = channelname + 'LVDT'
    elif 'Input' in channelname:
        datatype = 'IQ'
    elif 'InpI' in channelname:
        datatype = 'IQ'
    elif 'InpQ' in channelname:
        datatype = 'IQ'
    elif 'Disp' in channelname or 'Dissipation' in channelname:
        datatype = 'Dissipation'
    elif 'Time' in channelname:
        datatype = 'Time'
    elif 'Second' in channelname:
        datatype = 'Time'
    elif 'Temp' in channelname:
        # this is used by the force review, when you analyze as a funciton of 
        # StartHeadTemp, it asks this function what the units are.
        datatype = 'TemperatureParm'
    elif 'Velocity' in channelname:
        datatype = 'Velocity'
    elif 'SpringConstant' in channelname:
        datatype = 'K'
    elif 'SetPoint' in channelname:
        datatype = 'IQ'
    elif 'Cap' in channelname:
        datatype = 'S11_Log_Mag'
    elif 'Youngs' in channelname:
        datatype = 'Youngs'
    elif 'Modulus' in channelname:
        datatype = 'Youngs'
    elif 'Factor' in channelname or 'QFac' in channelname:
        datatype = 'Factor'
    elif 'L' in channelname and 'Tan' in channelname:
        datatype = 'Normalized'  # factor that is mostly 0-1

    output = ['nd']
    if datatype == 'Deflection':
        if dataTypeClass == 'Volts':
            output = ['V', 'm', 'N']
        elif dataTypeClass == 'Force':
            output = ['N', 'm', 'V']
        else:
            dataTypeClass = 'Meters'
            output = ['m', 'V', 'N']
    elif datatype == 'SurfaceEnergy':
        output = ["J/m^2"]
    elif datatype == 'Work':
        output = ['J']
    elif datatype in ['ZPiezo', 'YPiezo', 'XPiezo']:
        if dataTypeClass == 'Volts':
            output = ['V', 'm']
        else:
            dataTypeClass = 'Meters'
            output = ['m', 'V']
    elif datatype in ['XLVDT', 'YLVDT', 'ZLVDT']:
        if dataTypeClass == 'Volts':
            output = ['V', 'm']
        else:
            dataTypeClass = 'Meters'
            output = ['m', 'V']
    elif datatype == 'Amplitude':
        if dataTypeClass == 'Volts':
            output = ['V', 'm']
        else:
            dataTypeClass = 'Meters'
            output = ['m', 'V']
    elif datatype == 'Amplitude2':
        if dataTypeClass == 'Volts':
            output = ['V', 'm']
        else:
            dataTypeClass = 'Meters'
            output = ['m', 'V']
    elif datatype == 'Phase':
        dataTypeClass = 'Phase'
        output = ['deg']
    elif datatype == 'User':
        datatype += 'In'
    elif datatype == 'Lateral':
        if dataTypeClass == 'Meters':
            output = ['m', 'V']
        else:
            dataTypeClass = 'Volts'
            output = ['V', 'm']
    elif datatype in ['UserCalcB', 'UserCalc']:
        output = ['V']
        dataTypeClass = ''
    elif datatype == 'Current':
        output = ['A', 'V']
        dataTypeClass = 'Standard'
    elif datatype == 'Current2':
        output = ['A', 'V']
        dataTypeClass = 'Standard'
    elif datatype == 'S11_Log_Mag':
        output = ['db', 'db']
        dataTypeClass = 'Standard'
    elif datatype == 'Temperature':
        output = ['deg C']
        dataTypeClass = "Temperature"
    elif datatype == 'TipHeaterDrive':
        output = ['V']
        dataTypeClass = 'Volts'
    elif datatype == 'TipHeaterPower':
        output = ['M', 'V']
        dataTypeClass = 'Power'
    elif datatype == 'Youngs':
        output = ['Pa']
        dataTypeClass = 'Pressure'
    elif datatype in ['Volts', 'IQ']:
        output = ['V']
        dataTypeClass = 'Volts'
    elif datatype == 'Disipation':
        output = ['W']
        dataTypeClass = 'Power'
    elif datatype == 'Dissipation':
        output = ['V']
        dataTypeClass = 'Volts'
    elif datatype in ['Count2', 'Count1', 'Count0', 'Count']:
        output = [' ']
        dataTypeClass = 'Time'
    elif datatype == 'Time':
        output = ['Sec']
        dataTypeClass = 'Time'
    elif datatype == 'Frequency':
        output = ['Hz']
        dataTypeClass = 'Frequency'
    elif datatype == 'TemperatureParm':
        output = ['deg C']
        dataTypeClass = 'Time'
    elif datatype == 'Velocity':
        output = ['m/s']
        dataTypeClass = 'Meters'
    elif datatype == 'Invols':
        output = ['m/V']
        dataTypeClass = 'Meters'
    elif datatype == 'K':
        output = ['N/m']
        dataTypeClass = 'Meters'
    elif datatype == 'Factor':
        output = ['']
        dataTypeClass = 'Factor'
    elif datatype == 'Normalized':
        output = ['']
        dataTypeClass = 'Normalized'

    return output[0]
