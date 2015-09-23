import logging
import numpy
import pickle
import datetime
import h5py
from os import path
from mvpa2.misc.io import ColumnData
from mvpa2.datasets import dataset_wizard
from mvpa2.datasets.mri import fmri_dataset
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import permutation_test_score
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from neurometrics.neural_network import FeedForwardNeuralNetwork

def block_vote_score(y, y_pred):

    y = numpy.argmax(
        numpy.apply_along_axis(
            numpy.bincount, 
            1, 
            numpy.array(y.reshape(-1,6),dtype=int), 
            None, 
            4),
        axis=1)
    
    y_pred = numpy.argmax(
        numpy.apply_along_axis(
            numpy.bincount,
            1,
            numpy.array(y_pred.reshape(-1,6),dtype=int),
            None,
            4),
        axis=1)

    score = 1.0*sum(y==y_pred)/len(y)    

    return score

def do_session(session_dir,
               attr_file='attributes.txt',
               nifti_file='Timed_fixed.nii.gz',
               clf=None,
               scoring='accuracy',
               permutation_test = False):

    logger = logging.getLogger(__name__)

    logger.info(session_dir)

    logger.info('Loading attributes: {0}'.format(attr_file))
    attr = ColumnData(path.join(session_dir,attr_file))

    logger.info('Loading fmri dataset: {0}'.format(nifti_file))
    ds = fmri_dataset(samples = path.join(session_dir,nifti_file),
                      targets = attr.targets,
                      chunks = attr.runs)
    #samples = h5py.File(path.join(session,'study.hdf5'))['/3000/TimeSeries'][:].T.reshape(-1,3000)
    #ds = dataset_wizard(samples, targets = attr.targets, chunks = attr.runs)

    fs = SelectKBest(k=3000)

    fs.fit(ds.samples, ds.targets > 0)

    ds = ds[ds.targets > 0,:]

    ds = ds[:, fs.get_support()]

    logger.info('Configuring cross validation')
    cv = LeaveOneLabelOut(ds.chunks)

    logger.info('Beginning cross validation')
    scores = cross_val_score(clf,
                             ds.samples,
                             ds.targets,
                             cv = cv,
                             n_jobs = 8,
                             score_func = classification_report)

    if permutation_test:
        logger.info('Beginning permutation test')
        score, 
        permutation_scores, 
        pvalue = permutation_test_score(clf,
                                        ds.samples,
                                        ds.targets,
                                        cv = cv,
                                        n_jobs = 8,
                                        verbose = 50,
                                        scoring = 'accuracy')
        
    result = {}
    result['session_dir'] = session_dir
    result['datetime'] = datetime.datetime.now()
    result['attr_file'] = attr_file
    result['nifti_file'] = nifti_file
    result['clf'] = clf
    result['scoring'] = scoring
    result['scores'] = scores
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



 

