import logging
import numpy
import pickle
import datetime
import h5py
import operator
import math
from os import path
from mvpa2.misc.io import ColumnData
from mvpa2.datasets import dataset_wizard
from mvpa2.datasets import vstack, hstack
from mvpa2.datasets.base import AttrDataset
from mvpa2.datasets.mri import fmri_dataset
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import permutation_test_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from neurometrics.neural_network import FeedForwardNeuralNetwork


import nibabel.freesurfer.io

logger = logging.getLogger(__name__)

block_size = 18 #FIXME: figure out how to git rid of this

def vote(votes, classes=None, weights=None):
    if weights is None:
        weights = numpy.ones(votes.shape)
    if classes is None:
        classes = list(set(votes))
    return classes[numpy.argmax([weights[votes==c].sum() for c in classes])]

def block_vote_score(y_true, y_pred, block_size):
    classes = list(set(y_true))
    vote_true = [vote(v,classes) for v in y_true.reshape(-1,block_size)]
    vote_pred = [vote(v,classes) for v in y_pred.reshape(-1,block_size)]
    return accuracy_score(vote_true, vote_pred)

def block_probability_score(y_true, y_pred, y_proba, block_size):
    classes = list(set(y_true))
    vote_true = [vote(v,classes) for v in y_true.reshape(-1,block_size)]
    c = y_proba.reshape(len(classes),-1,block_size)
    vote_pred = numpy.argmax(c.sum(axis=2),axis=0)
    return accuracy_score(vote_true, vote_pred)

def score(clf, X, y):
    return {'report':classification_report(y,clf.predict(X)),
            'accuracy':accuracy_score(y,clf.predict(X)),
            'precision':precision_score(y,clf.predict(X),average='weighted'),
            'recall':recall_score(y,clf.predict(X),average='weighted'),
            'f1':f1_score(y,clf.predict(X),average='weighted'),
            'block_vote':block_vote_score(y,clf.predict(X),block_size),
            'block_proba':block_probability_score(y,clf.predict(X),clf.predict_proba(X),block_size) if clf.probability else None,
            'y':y,
            'predict':clf.predict(X),
            'predict_proba':clf.predict_proba(X) if clf.probability else None}

def train_score(clf, X, y, train, test, scoring):
    return scoring(clf.fit(X[train],y[train]),X[test],y[test])


import sklearn.base

def cross_val(clf, X, y, cv, scoring):
    #from IPython.parallel import Client
    #c = Client()
    #return c[:].map_sync(train_score, *zip(*[(clf, X, y, train, test, scoring) for train, test in cv]))
    return [train_score(sklearn.base.clone(clf),X,y,train,test,scoring) for train,test in cv]

def join_hemispheres(lhds, rhds):
    lhds.fa['hemi'] = ['lh']*lhds.nfeatures
    rhds.fa['hemi'] = ['rh']*rhds.nfeatures
    return hstack([lhds,rhds])

def join_datasets(datasets,a=None):

    # if a is None:
    #     a = list(set(reduce(operator.add,
    #                         [ds.a.keys() for ds in datasets])))
        
    # vds = []
    # for ds in datasets:
    #     for k in a:
    #         if k not in ds.sa:
    #             if k in ds.a:
    #                 ds.sa[k] = [ds.a[k]]*ds.nsamples
    #             else:
    #                 ds.sa[k] = None
    #     vds.append(ds)
    vds = datasets
        
    return vstack(vds)

def load_dataset(dataset_file):
    ds = AttrDataset.from_hdf5(dataset_file)
    return ds

def nifti_to_dataset(nifti_file, attr_file=None, annot_file=None, subject_id=None, session_id=None):

    logger.info('Loading fmri dataset: {}'.format(nifti_file))
    ds = fmri_dataset(samples = nifti_file)

    if attr_file is not None:
        logger.info('Loading attributes: {}'.format(attr_file))
        attr = ColumnData(attr_file)
        valid = min(ds.nsamples, attr.nrows)
        valid = int(valid/180)*180 #FIXME: ...
        print valid
        ds = ds[:valid,:]
        for k in attr.keys():
            ds.sa[k] = attr[k][:valid]

    if annot_file is not None:
        logger.info('Loading annotation: {}'.format(annot_file))
        annot = nibabel.freesurfer.io.read_annot(annot_file)
        ds.fa['annotation'] = [annot[2][i] for i in annot[0]]#FIXME: roi cannot be a fa

    if subject_id is not None:
        ds.sa['subject_id'] = [subject_id]*ds.nsamples

    if session_id is not None:
        ds.sa['session_id'] = [session_id]*ds.nsamples
        
    return ds

def do_falign(ds,
              alpha = 0.):

    ds.sa['chunks'] = ['{}:{}'.format(sid,scan)
                       for sid, scan
                       in zip(ds.sa.session_id,
                              ds.sa.run)]

    from mvpa2.mappers.detrend import PolyDetrendMapper

    detrender = PolyDetrendMapper(polyord = 1, chunks_attr='chunks')

    ds = ds.get_mapped(detrender)

    subjects = ds.sa['subject_id'].unique
    rois = ds.fa['annotation'].unique#FIXME: roi cannot be a fa
    hemis = ds.fa['hemi'].unique

    ds_lists = [[[ds[{'subject_id': [subject]},{'annotation': [roi], 'hemi': [hemi]}]
                  for subject in subjects]
                 for roi in rois]
                for hemi in hemis]

    from mvpa2.algorithms.hyperalignment import Hyperalignment
    from mvpa2.mappers.base import IdentityMapper

    def fa(s_list):
        try:
            return Hyperalignment(alpha=alpha)(s_list)
        except:
            logger.warning('Hyperalignment failed for {hemi} {roi}.'.format(hemi=s_list[0].fa.hemi[0],
                                                                            roi=s_list[0].fa.annotation[0]))
            logger.warning('Inserting identity mappers.')
            return [StaticProjectionMapper(numpy.eye(s.fa.attr_length)) for s in s_list]

    ha = dict(zip(hemis,[dict(zip(rois,[dict(zip(subjects,fa(s_list)))
                                        for s_list in r_list]))
                         for r_list in ds_lists]))

    return ha

def apply_falign(ds,
                 ha):

    subjects = ds.sa['subject_id'].unique
    rois = ds.fa['annotation'].unique#FIXME: roi cannot be a fa
    hemis = ds.fa['hemi'].unique

    rds = ds.copy()

    sds = []
    for subject in subjects:
        rds = []
        for roi in rois:
            hds = []
            for hemi in hemis:
                select = ({'subject_id': [subject]},{'annotation': [roi], 'hemi': [hemi]})
                mds = ha[hemi][roi][subject].forward(ds[select])
                mds.fa['annotation'] = ds[select].fa['annotation']
                mds.fa['hemi'] = ds[select].fa['hemi']
                hds.append(mds)
            rds.append(hstack(hds))
        sds.append(hstack(rds))
    return vstack(sds)


def do_session(ds,
               clf = SVC(kernel='linear', probability=True),
               scoring = score,
               targets = 'quantized_distance',
               n_jobs = 1,
               n_features = 3000,
               learning_curve = False,
               permutation_test = False):

    ds.sa['chunks'] = ['{}:{}'.format(sid,scan)
                       for sid, scan
                       in zip(ds.sa['session_id'],
                              ds.sa['run'])]

    ds.sa['targets'] = ds.sa[targets]

    #fixme: do wiener filter here

    from mvpa2.mappers.detrend import PolyDetrendMapper

    detrender = PolyDetrendMapper(polyord = 1, chunks_attr='chunks')

    ds = ds.get_mapped(detrender)

    ds = ds[numpy.logical_not(numpy.logical_or(ds.sa.move, ds.sa.cue)), :]

    if ds.nfeatures > n_features:
        fs = SelectKBest(k=n_features)
        fs.fit(ds.samples, ds.sa.search > 0)

    ds = ds[ds.sa.search > 0, :]

    if ds.nfeatures > n_features:
        ds = ds[:, fs.get_support()]

    logger.info('Configuring cross validation')
    cv = StratifiedKFold(ds.sa.quantized_distance, n_folds=6)#FIXME: make this a function parameter

    logger.info('Beginning cross validation')
    scores = cross_val(clf,
                       ds.samples,
                       ds.targets,
                       cv,
                       scoring)

    if learning_curve:
        from sklearn.learning_curve import learning_curve
        logger.info('Beginning learning curve analysis')

        train_sizes_abs, train_scores, test_scores = learning_curve(clf,
                                                                    ds.samples,
                                                                    ds.targets,
                                                                    n_jobs = n_jobs,
                                                                    verbose = 50,
                                                                    scoring = 'accuracy')
        

    if permutation_test:
        logger.info('Beginning permutation test')
        score, 
        permutation_scores, 
        pvalue = permutation_test_score(clf,
                                        ds.samples,
                                        ds.targets,
                                        cv = cv,
                                        n_jobs = n_jobs,
                                        verbose = 50,
                                        scoring = 'accuracy')
        
    result = {}
    result['datetime'] = datetime.datetime.now()
    if ds.nfeatures > n_features:
        result['fs'] = fs
    result['mapper'] = ds.mapper
    #result['clf'] = clf
    #result['cv'] = cv
    #result['scoring'] = scoring
    result['scores'] = scores
    if learning_curve:
        result['learning_curve'] = (train_sizes_abs,
                                    train_scores,
                                    test_scores)
    else:
        result['learning_curve'] = None
        
    if permutation_test:
        result['pvalue'] = pvalue;
    else:
        result['pvalue'] = None

    return result
       
#logging.basicConfig(level=logging.INFO)

#logger = logging.getLogger(__name__)

#sessions = ['S010614af',
#            'S010614drB',
#            'S040414drB',
#            'S040414jkA',
#	    'S042814vs',
#            'S050214af']
        
#logger.info('Configuring classifier')
#fs = SelectKBest(k = 3000)
#svc = SVC(kernel='linear')
#nn = FeedForwardNeuralNetwork()
#clf = Pipeline([('ANOVA',fs),('Classifier',svc)])               

#results = [do_session(session,clf=svc) for session in sessions]



 

