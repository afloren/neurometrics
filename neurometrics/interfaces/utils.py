from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, traits, File, TraitedSpec, DynamicTraitedSpec, Undefined, InputMultiPath
from nipype.utils.filemanip import split_filename

import neurometrics.utility
import neurometrics.ANOVA
import pickle
import gzip
import os
import stat
import string

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

class PerformMLInputSpec(BaseInterfaceInputSpec):
    nifti_file = File(desc='nifti file for ML to be performed on', exists=True, mandatory=True)
    attributes_file = File(desc='attributes.txt file containing target information', exists=True, mandatory=True)

class PerformMLOutputSpec(TraitedSpec):
    results_file = File(desc='pklz file containing results from ML', exists=True)
    
class PerformML(BaseInterface):
    input_spec = PerformMLInputSpec
    output_spec = PerformMLOutputSpec

    def _run_interface(self, runtime):
        results = neurometrics.ANOVA.do_session(self.inputs.attributes_file,
                                                self.inputs.nifti_file)
        with gzip.open(self._list_outputs()['results_file'],'wb') as f:
            pickle.dump(f, results, pickle.HIGHEST_PROTOCOL)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['results_file'] = os.path.abspath('results.pklz')#FIXME: should make this based on something
        return outputs

class PerformAcrossMLInputSpec(BaseInterfaceInputSpec):
    nifti_files = InputMultiPath(File(desc='nifti files for ML to be performed on', exists=True, mandatory=True))
    attributes_files = InputMultiPath(File(desc='attributes.txt files containing target information', exists=True, mandatory=True))

class PerformAcrossMLOutputSpec(TraitedSpec):
    results_file = File(desc='pklz file containing results from ML', exists=True)
    
class PerformAcrossML(BaseInterface):
    input_spec = PerformAcrossMLInputSpec
    output_spec = PerformAcrossMLOutputSpec

    def _run_interface(self, runtime):
        results = neurometrics.ANOVA.do_across_session(self.inputs.attributes_files,
                                                       self.inputs.nifti_files)
        with gzip.open(self._list_outputs()['results_file'],'wb') as f:
            pickle.dump(f, results, pickle.HIGHEST_PROTOCOL)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['results_file'] = os.path.abspath('results.pklz')#FIXME: should make this based on something
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
