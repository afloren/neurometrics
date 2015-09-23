#!/usr/bin/env python

import numpy as np
from mvpa2.misc.io.base import SampleAttributes
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.mappers.detrend import poly_detrend
from mvpa2.mappers.zscore import zscore
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.measures.base import CrossValidation

sa = SampleAttributes('attributes.txt')
ds = fmri_dataset(samples='func.feat/filtered_func_data.nii.gz',
                  targets=sa.targets,
                  chunks=sa.chunks)
poly_detrend(ds, polyord=1, chunks_attr='chunks')
zscore(ds, param_est=('targets',[0])
ds = ds[ds.sa.targets != 0]
clf = LinearCSVMC()
cvte = CrossValidation(clf, NFoldPartitioner(),
                       enable_ca=['stats'])
cv_results = cvte(ds)
sl = sphere_searchlight(cvte, readius=3, postproc=mean_sample())
sl_results = sl(ds)


