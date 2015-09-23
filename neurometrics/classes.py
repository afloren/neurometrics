from numpy import *
from os import path
import h5py

class Result:
    def __init__(self, name, distances, svm_confusion, nn_confusion):
        self.name = name
        self.distances = distances
        self.svm_confusion = svm_confusion
        self.nn_confusion = nn_confusion

class Session:
    def __init__(self, directory, 
                 file='study.hdf5', 
                 count='/Count', 
                 ts='/2000/TimeSeries', 
                 base_path='/export/data/mri/Neurometrics',
                 permute=False):
        self.directory = directory
        self.file = file
        self.base_path = base_path
        self.count = count
        self.ts = ts
        self.permute = permute
        self.results = []

    def getFullInputs(self):
        f = h5py.File(path.join(self.base_path,self.directory,self.file))
        inputs = array(f['/Full/TimeSeries'])
        inputs = inputs.reshape((prod(inputs.shape[:3]),)+inputs.shape[3:]).T
        inputs = inputs.reshape((prod(inputs.shape[:2]),)+inputs.shape[2:])
        inputs = inputs[:,all(~isnan(inputs),axis=0)]
        f.close()
        return inputs

    def getInputs(self):
        f = h5py.File(path.join(self.base_path,self.directory,self.file))
        inputs = array(f[self.ts]).T.reshape(-1,f[self.ts].shape[0])
        f.close()
        return inputs

    def getTargets(self):
        f = h5py.File(path.join(self.base_path,self.directory,self.file))
        targets = array(f[self.count])
        f.close()
        #if self.permute:
        #    targets = random.permutation(targets)
        return targets

class Subject:
    def __init__(self, name, sessions):
        self.name = name
        self.sessions = sessions

def countToTargets(count, classes=None):
    if classes is None:
        classes = unique(count)
    targets = zeros((len(count),len(classes)))
    for i,c in enumerate(classes):
        targets[count == c,i] = 1
    return targets
