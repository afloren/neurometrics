#! /usr/bin/env python

import nipype.interfaces.io as nio
import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, Directory, TraitedSpec, InputMultiPath
from nipype.pipeline.engine import Workflow

from sklearn import svm, metrics, feature_selection
from sklearn.cross_validation import StratifiedKFold, KFold

from classes import Subject, Session

import numpy as np

import pickle
from os import path

class SubWorkflow(Workflow):
    def __init__(self,name,input_fields=None,output_fields=None,**kwargs):
        Workflow.__init__(self,name=name,**kwargs)

        if input_fields:
            self.input_node = pe.Node(name = 'input',
                                      interface = util.IdentityInterface(fields=input_fields))
        if output_fields:
            self.output_node = pe.Node(name = 'output',
                                       interface = util.IdentityInterface(fields=output_fields))

class TrainSVMInputSpec(BaseInterfaceInputSpec):
    inputs = File(exists=True,desc='input values for training',mandatory=True)
    targets = File(exists=True,desc='target values for training',mandatory=True)
    indices = traits.Array(desc='indices for inputs and targets',mandatory=False)

class TrainSVMOutputSpec(TraitedSpec):
    svm = File(exists=True,desc='trained SVM')

class TrainSVM(BaseInterface):
    input_spec = TrainSVMInputSpec
    output_spec = TrainSVMOutputSpec

    def _run_interface(self, runtime):
        inputs = pickle.load(file(self.inputs.inputs,'r'))
        targets = pickle.load(file(self.inputs.targets,'r'))
        indices = self.inputs.indices
        if indices:
            inputs = inputs[indices]
            targets = targets[indices]
        svc = svm.SVC(C=1.0,kernel='linear')
        svc.fit(self.inputs.inputs,self.inputs.targets)
        pickle.dump(svc,file(path.abspath('svm.pkl'),'w'))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['svm'] = path.abspath('svm.pkl')
        return outputs

class TestSVMInputSpec(BaseInterfaceInputSpec):
    svm = File(exists=True,desc='SVM to be tested',mandatory=True)
    inputs = traits.Array(desc='input values to be tested',mandatory=True)
    targets = traits.Array(desc='target values to be tested',mandatory=True)
    indices = traits.Array(desc='indices for inputs and targets',mandatory=False)

class TestSVMOutputSpec(TraitedSpec):
    scores = File(exists=True,desc='SVM performance scores')

class TestSVM(BaseInterface):
    input_spec = TestSVMInputSpec
    output_spec = TestSVMOutputSpec

    def _run_interface(self, runtime):
        inputs =  self.inputs.inputs if self.inputs.indices is None else self.inputs.inputs[self.inputs.indices]
        targets = self.inputs.targets if self.inputs.indices is None else self.inputs.targets[self.inputs.indices]
        svc = pickle.load(file(self.inputs.svm,'r'))
        pred = svc.predict(self.inputs.inputs)
        m = metrics.confusion_matrix(self.inputs.targets,pred)
        pickle.dump(m,file(path.abspath('confusion.pkl'),'w'))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['scores'] = path.abspath('confusion.pkl')
        return outputs

class AggregateScoresInputSpec(BaseInterfaceInputSpec):
    scores = InputMultiPath(File(exists=True),desc='confusion matrices to be aggregated',mandatory=True)

class AggregateScoresOutputSpec(TraitedSpec):
    scores = File(exists=True,desc='aggregate confusion matrix')

class AggregateScores(BaseInterface):
    input_spec = AggregateScoresInputSpec
    output_spec = AggregateScoresOutputSpec
    
    def _run_interface(self, runtime):
        ms = np.array([pickle.load(file(f,'r')) for f in self.inputs.scores])
        m = ms.sum(axis=0)
        pickle.dump(m,file(path.abspath('confusion.pkl'),'w'))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['scores'] = path.abspath('confusion.pkl')
        return outputs

class PermuteArrayInputSpec(BaseInterfaceInputSpec):
    array = File(exists=True,desc='array to be permuted',mandatory=True)

class PermuteArrayOutputSpec(TraitedSpec):
    array = File(exists=True,desc='permuted array')

class PermuteArray(BaseInterface):
    input_spec = PermuteArrayInputSpec
    output_spec = PermuteArrayOutputSpec

    def _run_interface(self, runtime):
        a = pickle.load(file(self.inputs.array,'r'))
        b = np.random.permutation(a)
        pickle.dump(b,file(path.abspath('array.pkl'),'w'))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['array'] = path.abspath('array.pkl')
        return outputs

class ProcessSessionInputSpec(BaseInterfaceInputSpec):
    session_dir = Directory(exists=True,desc='session directory',mandatory=True)
    
class ProcessSessionOutputSpec(TraitedSpec):
    inputs = File(exists=True,desc='inputs for machine learning')
    targets = File(exists=True,desc='targets for machine learning')
    frames = File(exists=True,desc='frame indices')
    blocks = File(exists=True,desc='block indices')
    halfruns = File(exists=True,desc='halfrun indices')
    runs = File(exists=True,desc='run indices')
    frame_split = File(exists=True,desc='frame split')
    block_split = File(exists=True,desc='block split')
    halfrun_split = File(exists=True,desc='halfrun split')
    run_split = File(exists=True,desc='run split')

class ProcessSession(BaseInterface):
    input_spec = ProcessSessionInputSpec
    output_spec = ProcessSessionOutputSpec

    def _run_interface(self, runtime):
        run_length = 72
        block_length = 6
        
        session = Session(path.basename(self.inputs.session_dir),
                          base_path=path.dirname(self.inputs.session_dir))

        X = session.getInputs()
        y = session.getTargets()

        active = np.where(y > 0)
        X = X[active]
        y = y[active]

        num_frames = len(y)
        num_blocks = num_frames/block_length
        num_runs = num_frames/run_length
        num_halfruns = num_runs*2

        frames = np.arange(num_frames)
        blocks = np.arange(num_frames).reshape(-1,block_length)
        halfruns = np.arange(num_frames).reshape(-1,run_length/2)
        runs = np.arange(num_frames).reshape(-1,run_length)

        frame_split = StratifiedKFold(y[frames], num_frames/36)
        block_split = StratifiedKFold(y[blocks][:,0], num_blocks/6)
        halfrun_split = KFold(num_halfruns, num_halfruns)
        run_split = KFold(num_runs, num_runs)

        pickle.dump(X,file(path.abspath('inputs.pkl'),'w'))
        pickle.dump(y,file(path.abspath('targets.pkl'),'w'))
        pickle.dump(frames,file(path.abspath('frames.pkl'),'w'))
        pickle.dump(blocks,file(path.abspath('blocks.pkl'),'w'))
        pickle.dump(halfruns,file(path.abspath('halfruns.pkl'),'w'))
        pickle.dump(runs,file(path.abspath('runs.pkl'),'w'))
        pickle.dump(frame_split,file(path.abspath('frame_split.pkl'),'w'))
        pickle.dump(block_split,file(path.abspath('block_split.pkl'),'w'))
        pickle.dump(halfrun_split,file(path.abspath('halfrun_split.pkl'),'w'))
        pickle.dump(run_split,file(path.abspath('run_split.pkl'),'w'))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['inputs'] = path.abspath('inputs.pkl')
        outputs['targets'] = path.abspath('targets.pkl')
        outputs['frames'] = path.abspath('frames.pkl')
        outputs['blocks'] = path.abspath('blocks.pkl')
        outputs['halfruns'] = path.abspath('halfruns.pkl')
        outputs['runs'] = path.abspath('runs.pkl')
        outputs['frame_split'] = path.abspath('frame_split.pkl')
        outputs['block_split'] = path.abspath('block_split.pkl')
        outputs['halfrun_split'] = path.abspath('halfrun_split.pkl')
        outputs['run_split'] = path.abspath('run_split.pkl')
        return outputs

class ProcessFoldsInputSpec(BaseInterfaceInputSpec):
    inputs = File(exists=True,desc='inputs for machine learning',mandatory=True)
    targets = File(exists=True,desc='targets for machine learning',mandatory=True)
    indices = File(exists=True,desc='indices for inputs and targets',mandatory=True)
    folds = File(exists=True,desc='folds for crossvalidation',mandatory=True)

class ProcessFoldsOutputSpec(TraitedSpec):
    scores = File(exists=True,desc='aggregate confusion matrix for all folds')

class ProcessFolds(BaseInterface):
    input_spec = ProcessFoldsInputSpec
    output_spec = ProcessFoldsOutputSpec

    def _run_interface(self, runtime):
        inputs = pickle.load(file(self.inputs.inputs,'r'))
        targets = pickle.load(file(self.inputs.targets,'r'))
        indices = pickle.load(file(self.inputs.indices,'r'))
        folds = pickle.load(file(self.inputs.folds,'r'))
        ms = []
        for i,(train,test) in enumerate(folds):
            svc = svm.SVC(C=1.0,kernel='linear')
            svc.fit(inputs[indices[train].ravel()],targets[indices[train].ravel()])
            pickle.dump(svc,file(path.abspath('svc'+str(i)+'.pkl'),'w'))
            pred = svc.predict(inputs[indices[test].ravel()])
            m = metrics.confusion_matrix(targets[indices[test].ravel()],pred)
            pickle.dump(m,file(path.abspath('confusion'+str(i)+'.pkl'),'w')) 
            ms.append(m)
        ms = np.array(ms).sum(axis=0)
        pickle.dump(ms,file(path.abspath('confusion.pkl'),'w'))
        return runtime

    def _lit_outputs(self):
        outputs = self._outputs().get()
        outputs['scores'] = path.abspath('confusion.pkl')
        return outputs
    
class FoldWorkflow(SubWorkflow):
    def __init__(self,name,X,y,train,test):
        SubWorkflow.__init__(self,
                             name = name,
                             output_fields = ['scores'])
        
        self.train_node = pe.Node(name = 'TrainSVM',
                                  interface = TrainSVM())
        self.train_node.inputs.inputs = X
        self.train_node.inputs.targets = y
        self.train_node.inputs.indices = train

        self.test_node = pe.Node(name = 'TestSVM',
                                 interface = TestSVM())
        self.test_node.inputs.inputs = X
        self.test_node.inputs.targets = y
        self.test_node.inputs.indices = test

        self.connect(self.train_node,'svm',self.test_node,'svm')
        self.connect(self.test_node,'scores',self.output_node,'scores')
        
class CVWorkflow(SubWorkflow):
    def __init__(self,name):
        SubWorkflow.__init__(self,
                             name = name,
                             input_fields = ['inputs','targets','indices','folds'],
                             output_fields = ['scores'])

        self.cv_node = pe.Node(name = 'CV',
                               interface = ProcessFolds())

        self.connect(self.input_node,'inputs',self.cv_node,'inputs')
        self.connect(self.input_node,'targets',self.cv_node,'targets')
        self.connect(self.input_node,'indices',self.cv_node,'indices')
        self.connect(self.input_node,'folds',self.cv_node,'folds')
        self.connect(self.cv_node,'scores',self.output_node,'scores')

class SessionWorkflow(SubWorkflow):
    def __init__(self,session):
        SubWorkflow.__init__(self,
                             name = session.directory,
                             input_fields = ['session_dir'],
                             output_fields = ['frame_scores','block_scores','halfrun_scores','run_scores'])

        #self.input_node.session_dir = path.join(session.base_path,session.directory)
            

        self.process_node = pe.Node(name = 'Process',
                                    interface = ProcessSession())

        self.process_node.inputs.session_dir = path.join(session.base_path,session.directory)

        #self.connect(self.input_node,'session_dir',self.process_node,'session_dir')


        if session.permute:
            self.permute_node = pe.Node(name='permute',
                                        interface = PermuteArray())
            self.connect(self.process_node,'targets',self.permute_node,'array')

        self.cv_workflows = dict()

        for name in ('frame','block','halfrun','run'):
            wf = CVWorkflow(name)
            self.cv_workflows[name] = wf
            self.connect(self.process_node,'inputs',wf,'input.inputs')
            if session.permute:
                self.connect(self.permute_node,'array',wf,'input.targets')
            else:
                self.connect(self.process_node,'targets',wf,'input.targets')
            self.connect(self.process_node,name+'s',wf,'input.indices')
            self.connect(self.process_node,name+'_split',wf,'input.folds')
            self.connect(wf,'output.scores',self.output_node,name+'_scores')

class SubjectWorkflow(SubWorkflow):
    def __init__(self,subject):
        SubWorkflow.__init__(self,
                             name = subject.name,
                             output_fields = ['frame_scores','block_scores','halfrun_scores','run_scores'])

        self.session_workflows = [SessionWorkflow(s) for s in subject.sessions]

        self.merge_nodes = dict()
        self.aggregate_nodes = dict()

        for name in ('frame','block','halfrun','run'):
            mn = pe.Node(name = name+'_merge',interface = util.Merge(len(subject.sessions)))
            self.merge_nodes[name] = mn
            an = pe.Node(name = name+'_aggregate',interface = AggregateScores())
            self.aggregate_nodes[name] = an
            for i,sw in enumerate(self.session_workflows):
                self.connect(sw,'output.'+name+'_scores',mn,'in'+str(i+1))
            self.connect(mn,'out',an,'scores')
            self.connect(an,'scores',self.output_node,name+'_scores')

class Analysis1(SubWorkflow):
    def __init__(self,name='Analysis1',session_dir='/export/data/mri/Neurometrics',permute=False,**kwargs):
        SubWorkflow.__init__(self,
                             name = name,
                             output_fields = ['frame_scores','block_scores','halfrun_scores','run_scores'],
                             **kwargs)

        subjects = [Subject('a',[Session('020211el',base_path=session_dir,permute=permute),Session('021611el',base_path=session_dir,permute=permute)]),
                    Subject('b',[Session('020511bn',base_path=session_dir,permute=permute),Session('021211bn',base_path=session_dir,permute=permute)]),
                    Subject('c',[Session('021911ar',base_path=session_dir,permute=permute),Session('090210ar1',base_path=session_dir,permute=permute)]),
                    Subject('d',[Session('080612af',base_path=session_dir,permute=permute),Session('091912af',base_path=session_dir,permute=permute)]),
                    Subject('e',[Session('101812rm',base_path=session_dir,permute=permute),Session('011513rm',base_path=session_dir,permute=permute)])]

        self.subject_workflows = [SubjectWorkflow(u) for u in subjects]

        self.merge_nodes = dict()
        self.aggregate_nodes = dict()

        for name in ('frame','block','halfrun','run'):
            mn = pe.Node(name = name+'_merge',interface = util.Merge(len(subjects)))
            self.merge_nodes[name] = mn
            an = pe.Node(name = name+'_aggregate',interface = AggregateScores())
            for i,uw in enumerate(self.subject_workflows):
                self.connect(uw,'output.'+name+'_scores',mn,'in'+str(i+1))
            self.connect(mn,'out',an,'scores')
            self.connect(an,'scores',self.output_node,name+'_scores')

class PermutationAnalysis(Workflow):
    def __init__(self,name='PermutationAnalysis',session_dir='/export/data/mri/Neurometrics',n=2000,**kwargs):
        Workflow.__init__(self,name=name,**kwargs)

        self.permutation_workflows = [Analysis1('p{0}'.format(i),session_dir,permute=True) for i in range(n)]
            
        self.datasink = pe.Node(name = 'DataSink',interface = nio.DataSink())
        self.datasink.inputs.base_directory = session_dir
        self.datasink.inputs.container = 'ptest'

        for pw in self.permutation_workflows:
            self.connect(pw,'output.frame_scores',self.datasink,pw.name+'.frame')
            self.connect(pw,'output.block_scores',self.datasink,pw.name+'.block')
            self.connect(pw,'output.halfrun_scores',self.datasink,pw.name+'.halfrun')
            self.connect(pw,'output.run_scores',self.datasink,pw.name+'.run')

