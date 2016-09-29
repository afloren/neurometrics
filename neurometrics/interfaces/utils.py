from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec, DynamicTraitedSpec, Undefined, InputMultiPath
from nipype.utils.filemanip import split_filename

import neurometrics.utility
import neurometrics.ANOVA
import pickle
import gzip
import os
import stat
import string
import numpy as np

import nilearn
import nilearn.image

class ExtractVolumeInputSpec(BaseInterfaceInputSpec):
    in_file = File(desc='input volume', exists=True, mandatory=True)
    index = traits.Int(desc='index of volume to extract', mandatory=True)

class ExtractVolumeOutputSpec(TraitedSpec):
    out_file = File(desc='extracted volume', exists=True)

class ExtractVolume(BaseInterface):
    input_spec = ExtractVolumeInputSpec
    output_spec = ExtractVolumeOutputSpec

    def _run_interface(self, runtime):
        nim = nilearn.image.index_img(self.inputs.in_file, self.inputs.index)
        nim.to_filename(self._list_outputs()['out_file'])
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        base = os.path.splitext(os.path.basename(fname))[0]
        outputs['out_file'] = os.path.abspath(base + '_ref.nii.gz')
        return outputs

class LtaToXfmInputSpec(BaseInterfaceInputSpec):
    in_file = File(desc='input lta file', exists=True, mandatory=True)
    out_file = File(desc='output xfm file', hash_files=False, name_source=['in_file'], name_template='%s.xfm')#FIXME: redundant with _list_outputs
    
class LtaToXfmOutputSpec(TraitedSpec):
    out_file = File(desc='output xfm file', exists=True)

class LtaToXfm(BaseInterface):
    input_spec = LtaToXfmInputSpec
    output_spec = LtaToXfmOutputSpec

    def _run_interface(self, runtime):
        arr = neurometrics.utility.load_lta(self.inputs.in_file)
        neurometrics.utility.save_xfm(arr[:3,:],self._list_outputs()['out_file'])        
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.in_file
        base = os.path.splitext(os.path.basename(fname))[0]
        outputs["out_file"] = os.path.abspath(base + '.xfm')#FIXME: should check if inputs.out_file is defined
        return outputs

class NiftiToDatasetInputSpec(BaseInterfaceInputSpec):
    nifti_file = File(desc='nifti file to convert to ML dataset format', exists=True, mandatory=True)
    attributes_file = File(desc='attribute file containing information for ML', exists=True)
    annot_file = File(desc='annotation file containing parcelation information', exists=True)
    subject_id = traits.String(desc='unique subject identifier')
    session_id = traits.String(desc='unique session identifier')
    ds_file = traits.String('dataset.hdf5', desc='name of ds file', usedefault=True) 

class NiftiToDatasetOutputSpec(TraitedSpec):
    ds_file = File(desc='output file in ML dataset format', exists=True)
    
class NiftiToDataset(BaseInterface):
    input_spec = NiftiToDatasetInputSpec
    output_spec = NiftiToDatasetOutputSpec

    def _run_interface(self, runtime):
        ds = neurometrics.ANOVA.nifti_to_dataset(self.inputs.nifti_file,
                                                 self.inputs.attributes_file,
                                                 self.inputs.annot_file if self.inputs.annot_file is not Undefined else None,
                                                 self.inputs.subject_id,
                                                 self.inputs.session_id)
        ds.save(self._list_outputs()['ds_file'])
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['ds_file'] = os.path.abspath(self.inputs.ds_file)
        return outputs

class JoinDatasetsInputSpec(BaseInterfaceInputSpec):
    input_datasets = InputMultiPath(File(desc='datasets to be joined', exists=True, mandatory=True))
    join_hemispheres = traits.Bool(desc='whether we are joining hemispheres or not')
    joined_dataset = traits.String('dataset.hdf5', desc='name of joined dataset file', usedefault=True) 

class JoinDatasetsOutputSpec(TraitedSpec):
    joined_dataset = File(desc='joined dataset', exists=True)
    
class JoinDatasets(BaseInterface):
    input_spec = JoinDatasetsInputSpec
    output_spec = JoinDatasetsOutputSpec

    def _run_interface(self, runtime):
        datasets = [neurometrics.ANOVA.load_dataset(d)
                    for d in self.inputs.input_datasets]
        if self.inputs.join_hemispheres:
            assert(len(datasets) == 2)
            ds = neurometrics.ANOVA.join_hemispheres(datasets[0],
                                                     datasets[1])
        else:
            ds = neurometrics.ANOVA.join_datasets(datasets)
        ds.save(self._list_outputs()['joined_dataset'])
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['joined_dataset'] = os.path.abspath(self.inputs.joined_dataset)
        return outputs
    
class PerformMLInputSpec(BaseInterfaceInputSpec):
    ds_file = File(desc='dataset file for ML to be performed on', exists=True, mandatory=True)
    classifier = traits.Any(None)#TODO: make this a classifier object
    scoring = traits.Any(None)#TODO: make this a scoring object
    targets = traits.Any(None)#TODO: make this a string
    learning_curve = traits.Bool(False, desc='whether training curve analysis will be performed')
    
class PerformMLOutputSpec(TraitedSpec):
    results_file = File(desc='pklz file containing results from ML', exists=True)
    
class PerformML(BaseInterface):
    input_spec = PerformMLInputSpec
    output_spec = PerformMLOutputSpec

    def _run_interface(self, runtime):
        ds = neurometrics.ANOVA.load_dataset(self.inputs.ds_file)
        
        kwargs = {}
        if self.inputs.classifier:
            kwargs['clf'] = self.inputs.classifier
        if self.inputs.scoring:
            kwargs['scoring'] = self.inputs.scoring
        if self.inputs.targets:
            kwargs['targets'] = self.inputs.targets
        if self.inputs.learning_curve:
            kwargs['learning_curve'] = self.inputs.learning_curve
                
        results = neurometrics.ANOVA.do_session(ds,
                                                **kwargs)

        with gzip.open(self._list_outputs()['results_file'],'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['results_file'] = os.path.abspath('results.pklz')#FIXME: should make this based on something
        return outputs

class PerformFAInputSpec(BaseInterfaceInputSpec):
    ds_file = File(desc='dataset file for FA to be performed on', exists=True, mandatory=True)

class PerformFAOutputSpec(TraitedSpec):
    out_file = File(desc='pklz file containing dictionary of FA mappers', exists=True)

class PerformFA(BaseInterface):
    input_spec = PerformFAInputSpec
    output_spec = PerformFAOutputSpec

    def _run_interface(self, runtime):
        ds = neurometrics.ANOVA.load_dataset(self.inputs.ds_file)
        ha = neurometrics.ANOVA.do_falign(ds,
                                          self.inputs.classifier,
                                          self.inputs.scoring,
                                          self.inputs.targets)
        with gzip.open(self._list_outputs()['out_file'],'wb') as f:
            pickle.dump(ha, f, pickle.HIGHEST_PROTOCOL)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath('ha.pklz')#FIXME: should make this based on something
        return outputs

class ApplyFAInputSpec(BaseInterfaceInputSpec):
    ds_file = File(desc='dataset file to apply FA to', exists=True, mandatory=True)
    ha_file = File(desc='pklz file containing dictionary of FA mappers', exists=True, mandatory=True)

class ApplyFAOutputSpec(TraitedSpec):
    out_file = File(desc='FA transformed dataset', exists=True)

class ApplyFA(BaseInterface):
    input_spec = ApplyFAInputSpec
    output_spec = ApplyFAOutputSpec

    def _run_interface(self, runtime):
        ds = neurometrics.ANOVA.load_dataset(self.inputs.ds_file)
        with gzip.open(self.inputs.ha_file,'rb') as f:
            ha = pickle.load(f)
        fads = neurometrics.ANOVA.apply_falign(ds,ha)
        fads.save(self._list_outputs()['out_file'])
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = os.path.abspath('fa.hdf5')#FIXME: should make this based on something
        return outputs

class SummarizeResultsInputSpec(BaseInterfaceInputSpec):
    ids = InputMultiPath(desc='unique identifier for each result', mandatory=True)
    results_files = InputMultiPath(File(desc='results files to be summarized', exists=True, mandatory=True))
    
class SummarizeResultsOutputSpec(TraitedSpec):
    summary_file = File(desc='text file containing summary of ML results', exists=True)

class SummarizeResults(BaseInterface):
    input_spec = SummarizeResultsInputSpec
    output_spec = SummarizeResultsOutputSpec

    def _run_interface(self, runtime):
        ids = self.inputs.ids
        
        results = []
        for rfile in self.inputs.results_files:
            with gzip.open(rfile) as f:
                results.append(pickle.load(f))

        #keys = ['accuracy','recall','f1','block_vote','block_proba']
        
        if all(isinstance(v,dict) for r in results for v in r['scores']):
            keys = set.intersection(*[set(d.keys()) for r in results for d in r['scores']])
        else:
            keys = None
                
        with open(self._list_outputs()['summary_file'],'w') as f:
            for i,r in zip(ids,results):
                f.write('ID: {}\n'.format(i))
                f.write('Average scores\n')
                if keys:
                    for k in keys:
                        f.write('{}\n'.format(k))
                        try:
                            val = np.mean([v[k] for v in r['scores']])
                            f.write('{}\n'.format(val))
                        except Exception as e:
                            f.write('{}\n'.format(e))
                else:
                    try:
                        val = np.mean(r['scores'])
                        f.write('{}\n'.format(val))
                    except Exception as e:
                        f.write('{}\n'.format(e))
                f.write('Fold scores\n')
                for j,v in enumerate(r['scores']):
                    f.write('fold: {}\n'.format(j))
                    if keys:
                        for k in keys:
                            f.write('{}\n'.format(k))
                            f.write('{}\n'.format(v[k]))
                    else:
                        f.write('{}\n'.format(v))
                if keys:
                    f.write('Other scores\n')
                    for j,v in enumerate(r['scores']):
                        f.write('fold: {}\n'.format(j))
                        for k in v.keys():
                            if k not in keys:
                                f.write('{}\n'.format(k))
                                f.write('{}\n'.format(v[k]))
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['summary_file'] = os.path.abspath('results.txt')
        return outputs

class WriteFileInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    out_file = File(desc='output file', hash_files=False)

class WriteFileOutputSpec(TraitedSpec):
    out_file = File(desc='output file', exists=True)
    
class WriteFile(BaseInterface):
    input_spec = WriteFileInputSpec
    output_spec = WriteFileOutputSpec

    def __init__(self, template, **kwargs):
        super(WriteFile, self).__init__(**kwargs)

        infields = []
        for _, field_name, _, _ in string.Formatter().parse(template):
            if field_name is not None and field_name not in infields:
                infields.append(field_name)

        self._infields = infields
        self._template = template

        undefined_traits = {}
        for field in infields:
            self.inputs.add_trait(field, traits.Any)
            undefined_traits[field] = Undefined
        self.inputs.trait_set(trait_change_notify=False, **undefined_traits)
    
    def _run_interface(self, runtime):
        outputs = self._list_outputs()
        with open(outputs['out_file'],'w') as f:
            f.write(self._template.format(**self.inputs.__dict__))
        st_mode = os.stat(outputs['out_file']).st_mode
        os.chmod(outputs['out_file'],
                 st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return runtime

    def _gen_filename(self):
        return 'out_file'
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        if self.inputs.out_file:
            outputs['out_file'] = self.inputs.out_file
        else:
            outputs['out_file'] = os.path.abspath(self._gen_filename())
        return outputs

class AlignmentQA(WriteFile):
    qa_template = ('#!/bin/bash -i\n'
                   'freeview -v {target_file} -v {source_file}:reg={reg_file}\n')
    
    def __init__(self, **kwargs):
        super(AlignmentQA, self).__init__(AlignmentQA.qa_template, **kwargs)

    def _gen_filename(self):
        return 'alignment-qa.sh'

