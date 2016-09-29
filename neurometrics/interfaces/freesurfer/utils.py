import os
import re
from nipype.utils.filemanip import fname_presuffix, split_filename

from nipype.interfaces.freesurfer.base import FSCommand, FSTraitedSpec
from nipype.interfaces.base import TraitedSpec, File, traits, OutputMultiPath, isdefined, CommandLine, CommandLineInputSpec

filemap = dict(cor='cor', mgh='mgh', mgz='mgz', minc='mnc',
               afni='brik', brik='brik', bshort='bshort',
               spm='img', analyze='img', analyze4d='img',
               bfloat='bfloat', nifti1='img', nii='nii',
               niigz='nii.gz')

filetypes = ['cor', 'mgh', 'mgz', 'minc', 'analyze',
             'analyze4d', 'spm', 'afni', 'brik', 'bshort',
             'bfloat', 'sdt', 'outline', 'otl', 'gdf',
             'nifti1', 'nii', 'niigz']


class LabelToVolumeInputSpec(FSTraitedSpec):
    # --label labelid <--label labelid>
    label = traits.File(desc='labelid <--label labelid>',
                        argstr='--label %s', exists=True)
    # --annot annotfile : surface annotation file
    annot = traits.File(desc='annotfile : surface annotation file',
                        argstr='--annot %s', exists=True)
    # --seg   segpath : segmentation
    seg = traits.File(desc='segpath : segmentation',
                      argstr='--seg %s', exists=True)
    # --aparc+aseg  : use aparc+aseg.mgz in subjectdir as seg
    aparc = traits.Bool(desc=': use aparc+asec.mgz in sibjectdir as seg',
                        argstr='--aparc+aseg')#fixme xor seg and requires subject

    # --temp tempvolid : output template volume
    temp = traits.File(desc='tempvolid : output template volume',
                       argstr='--temp %s', exists=True)

    # --reg regmat : VolXYZ = R*LabelXYZ
    reg = traits.File(desc='regmat : VolXYZ = R*LabelXYZ')
    # --regheader volid : label template volume (needed with --label or --annot)
    #             for --seg, use the segmentation volume
    regheader = traits.File(desc='volid : label template volume (needed with --label or --anot) for --seg, use the segmentation volume',
                            argstr='--regheader %s', exists=True)#fixme xor with seg (maybe require label/anot)

    # --identity : set R=I
    identity = traits.Bool(desc='set R=I',
                           argstr='--identity')
    # --invertmtx : Invert the registration matrix
    invertmtx = traits.Bool(desc='Invert the registration matrix',
                            argstr='--invertmtx')

    # --fillthresh thresh : between 0 and 1 (def 0)
    fillthresh = traits.Float(desc='thresh : between 0 and 1 (def 0)',
                              argstr='--fillthresh %s')
    # --labvoxvol voxvol : volume of each label point (def 1mm3)
    labvoxvol = traits.Str(desc='voxvol : volume of each label point (def 1mm3)',
                           argstr='--labvoxvol %s')
    # --proj type start stop delta
    #fixme how to implement?

    # --subject subjectid : needed with --proj or --annot
    subject = traits.Str()
    # --hemi hemi : needed with --proj or --annot
    # --surf surface  : use surface instead of white

    # --o volid : output volume
    # --hits hitvolid : each frame is nhits for a label
    # --label-stat statvol : map the label stats field into the vol
    # --stat-thresh thresh : only use label point where stat > thresh
    # --offset k : add k to segmentation numbers (good when 0 should not be ignored)

    # --native-vox2ras : use native vox2ras xform instead of tkregister-style

class LabelToVolumeOutputSpec(TraitedSpec):
    pass

class LabelToVolume(FSCommand):
    _cmd = 'mri_label2vol'
    input_spec = LabelToVolumeInputSpec
    output_spec = LabelToVolumeOutputSpec

