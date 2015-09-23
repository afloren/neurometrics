#!/usr/bin/env python

from numpy import *

def HarmonicPower(array, nCycles, nHarmonics=4, axis=0):
    array = rollaxis(array,axis)

    ft = fft.fft(array,axis=0)
    nSpec = 1 + floor(ft.shape[0]/2.0)
    ft = ft[:nSpec]
    
    pwr = power(abs(ft),2)
    totalPwr = pwr.sum(axis=0)
    scaledPwr = 100 * pwr / totalPwr

    harms = nCycles * arange(1,nHarmonics+1)
    
    hp = scaledPwr[harms].sum(axis=0)

    return hp
    
