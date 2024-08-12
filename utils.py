import sys
import numpy as np
# Somehow here, we need to add this directory...
from os.path import dirname
sys.path.append('/home/kutnyakd/sed_old/')

from sed import SedProcessor
import sed
import copy



"""
    Parameters determined from notebook '2023_03_14' for the spin-integrated branch

paramsStraight = {'offset': 45711.422845691384, 'coeffs': np.array([-4.39448092e-08,  5.04262934e-03]), 'Tmat': np.array([[ 3.22082743e+08,  3.39679359e+03],
       [ 2.33215630e+08,  2.43486974e+03],
       [ 1.25282328e+08,  1.29258517e+03],
       [-1.55968048e+08, -1.56312625e+03],
       [-3.51475998e+08, -3.45691383e+03]]), 'bvec': np.array([ 3.,  2.,  1., -1., -2.]), 'axis': -141.65648783097132, 'E0': -141.65648783097132, 'refid': 3}
"""

"""
    Parameters determined from notebook '2023_03_23'
"""

paramsStraight = {'offset': 33156.3126252505, 'coeffs': np.array([-9.79653384e-08,  9.61013178e-03]), 'Tmat': np.array([[ 6.74495283e+07,  1.00200401e+03],
       [ 4.40706061e+07,  6.51302605e+02],
       [ 2.38357476e+07,  3.50701403e+02],
       [-2.40817306e+07, -3.50701403e+02]]), 'bvec': np.array([ 3.,  2.,  1., -1.]), 'axis': -213.9608886309967, 'E0': -213.9608886309967, 'refid': 3}


"""
    Parameters determined from notebook '2023_03_14' for the spin branch
"""
parametersSpin = {'offset': 78480, 'coeffs': np.array([-1.37972371e-08,  2.76823747e-03]), 'Tmat': np.array([[923795200,      5680],
       [671700700,      4090],
       [334283100,      2010]]), 'bvec': np.array([3., 2., 1.]), 'axis': -135.25033778652647, 'E0': -135.25033778652647, 'refid': 3}



"""
    Parameters determined from notebook '2023_03_24' for the spin branch
"""

parametersSpin = {'offset': 102700, 'coeffs': np.array([0.00041132]), 'Tmat': np.array([[ 3680],[ 2420], [ 1180],[-1170]]), 'bvec': np.array([ 1.5,  1. ,  0.5, -0.5]), 'axis': -43.75673846237156, 'E0': -43.75673846237156, 'refid': 3}


 
    
def tablePerTrain(sp):
    # Generate the processor per bunch
    sp_per_train = copy.deepcopy(sp)
    sp_per_train._dataframe = sp_per_train._dataframe.drop_duplicates(subset='trainId').drop(
                    columns=['dldTime', 'dldPosX', 'dldPosY'])
    return sp_per_train
    
    
def calibrateEnergyNoSpin(sp, mode = 'poly', flipSign=True):#, bias=28):
    """
    Create a new axis 'energy' from the dldTime using the energy calibration for the spin-integrated branch.
    There are two modes:
        'linear': just use a linear scaling
    """
    
    if (mode == 'poly'):
        sp._dataframe = sp.ec.append_energy_axis(df=sp._dataframe,
                                             tof_column = 'dldTime',
                                             energy_column = 'energy',

                                             a=paramsStraight['coeffs'],
                                             E0=paramsStraight['E0'])
    else:
        print('mode ' + mode + 'not supported')
        return
    
    if (flipSign):
        sp._dataframe['energy'] = - sp._dataframe['energy'] - 2.0
        
    #sp._dataframe['energy'] = sp._dataframe['energy']-(31-bias)   

def calibrateEnergySpin(sp, mode = 'poly', flipSign=True, bias=30):
    """
    Create a new axis 'energy' from the dldTime using the energy calibration for the spin-integrated branch.
    There are two modes:
        'linear': just use a linear scaling
    """
    
    if (mode == 'poly'):
        sp._dataframe = sp.ec.append_energy_axis(df=sp._dataframe,
                                             tof_column = 'dldTime',
                                             energy_column = 'energy',

                                             a=parametersSpin['coeffs'],
                                             E0=parametersSpin['E0'])
    else:
        print('mode ' + mode + 'not supported')
        return
    
    if (flipSign):
        sp._dataframe['energy'] = - sp._dataframe['energy']
    
    sp._dataframe['energy'] = sp._dataframe['energy']-(30-bias)   
        