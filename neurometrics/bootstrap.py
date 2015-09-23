#! /usr/bin/env python

from random import random
from numpy import array
from numpy import apply_along_axis as along
from scipy.stats import scoreatpercentile as sap

def samplewr(arr,n=None):
    if n is None:
        n = len(arr)
    return array([arr[int(random()*len(arr))] for i in xrange(n)])

def bootstrap(arr,estimator,a,n,axis=None):
    if axis is None:
        arr = arr.ravel()
        axis = 0 
    t = [along(estimator,axis,along(samplewr,axis,arr)) for i in xrange(n)]
    return array((along(lambda x: sap(x,a),0,t),
                  along(lambda x: sap(x,1-a),0,t))).T #can simplify on newer scipy package
    
