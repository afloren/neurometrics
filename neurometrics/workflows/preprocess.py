import nipype.interfaces.io as nio
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.dcmstack as ds
import nipype.interfaces.nipy as nipy
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

import neurometrics.interfaces.utils as nmutil

from mvpa2.misc.io import ColumnData

def create_extract_inplane_workflow(name='extract_inplane',
                                    templates={'inplane':'Raw/Anatomy/Inplane/'},
                                    format_string='inplane'):

    extract_inplane = pe.Workflow(name=name)

    inputs = pe.Node(interface=util.IdentityInterface(fields=['session_dir']), name='inputs')
    get_inplane = pe.Node(interface=nio.SelectFiles(templates), name='get_inplane')
    inplane_to_nii = pe.Node(interface=ds.DcmStack(), name='inplane_to_nii')
    inplane_to_nii.inputs.embed_meta = True
    rename_inplane = pe.Node(interface=util.Rename(format_string), name='rename_inplane')
    rename_inplane.inputs.keep_ext = True
    outputs = pe.Node(interface=util.IdentityInterface(fields=['out_file']), name='outputs')
    
    extract_inplane.connect(inputs,'session_dir',get_inplane,'base_directory')
    extract_inplane.connect(get_inplane,'inplane',inplane_to_nii,'dicom_files')
    extract_inplane.connect(inplane_to_nii,'out_file',rename_inplane,'in_file')
    extract_inplane.connect(rename_inplane,'out_file',outputs,'out_file')
    
    return extract_inplane

def create_extract_functional_workflow(name='extract_functional',
                                       templates={'functional':'Raw/Functional/Scan_{scan}/'},
                                       format_string='f%(scan)d'):

    extract_functional = pe.Workflow(name=name)

    inputs = pe.Node(interface=util.IdentityInterface(fields=['session_dir','scan']), name='inputs')
    get_functional = pe.Node(interface=nio.SelectFiles(templates), name='get_functional')
    functional_to_nii = pe.Node(interface=ds.DcmStack(), name='functional_to_nii')
    functional_to_nii.inputs.embed_meta = True
    rename_functional = pe.Node(interface=util.Rename(format_string), name='rename_functional')
    rename_functional.inputs.keep_ext = True
    outputs = pe.Node(interface=util.IdentityInterface(fields=['out_file']), name='outputs')
    
    extract_functional.connect(inputs,'session_dir',get_functional,'base_directory')
    extract_functional.connect(inputs,'scan',get_functional,'scan')
    extract_functional.connect(get_functional,'functional',functional_to_nii,'dicom_files')
    extract_functional.connect(functional_to_nii,'out_file',rename_functional,'in_file')
    extract_functional.connect(inputs,'scan',rename_functional,'scan')
    extract_functional.connect(rename_functional,'out_file', outputs, 'out_file')
    
    return extract_functional

def create_align_to_anatomy_workflow(name='align_to_anatomy',
                                     format_string = 'inplane_to_anatomy'):
    
    align_to_anatomy = pe.Workflow(name=name)

    inputs = pe.Node(interface=util.IdentityInterface(fields=['inplane_file', 'anatomy_file']), name='inputs') 
    strip = pe.Node(interface=fs.ReconAll(), name='strip')#FIXME: reconall interface barfs if rerun
    strip.inputs.directive = 'autorecon1'
    strip.inputs.flags = '-nowsgcaatlas'

    register = pe.Node(interface=fs.RobustRegister(), name='register')
    register.inputs.auto_sens = True
    register.inputs.init_orient = True
    convert_xfm = pe.Node(interface=nmutil.LtaToXfm(), name='convert_xfm')
    rename_xfm = pe.Node(interface=util.Rename(format_string), name='rename_xfm')
    rename_xfm.inputs.keep_ext = True
    outputs = pe.Node(interface=util.IdentityInterface(fields=['xfm_file','strip_file']), name='outputs')

    align_to_anatomy.connect(inputs, 'inplane_file', strip, 'T1_files')
    align_to_anatomy.connect(strip, 'brainmask', register, 'source_file')
    align_to_anatomy.connect(inputs, 'anatomy_file', register, 'target_file')
    align_to_anatomy.connect(register, 'out_reg_file', convert_xfm, 'in_file')
    align_to_anatomy.connect(convert_xfm, 'out_file',rename_xfm,'in_file')
    align_to_anatomy.connect(rename_xfm, 'out_file', outputs, 'xfm_file')
    align_to_anatomy.connect(strip, 'brainmask', outputs, 'strip_file')
    
    return align_to_anatomy

#assumptions: slice acquisition timing is the same for all frames, slices are organized along dimension 2
def create_within_run_align_workflow(name='within_run_align', slice_timing_correction=True):

    within_run_align = pe.Workflow(name=name)

    inputs = pe.Node(interface=util.IdentityInterface(fields=['in_file']), name='inputs')


    if slice_timing_correction:
        get_meta = pe.Node(interface=ds.LookupMeta(), name='get_meta')
        get_meta.inputs.meta_keys = {'RepetitionTime':'tr',
                                     'CsaImage.MosaicRefAcqTimes':'slice_times'}

        select_slice_times = pe.Node(interface=util.Select(), name='select_slice_times')
        select_slice_times.inputs.index = [0]

    space_time_align = pe.Node(interface=nipy.SpaceTimeRealigner(), name='space_time_align')

    if slice_timing_correction:
        space_time_align.inputs.slice_info = 2

    outputs = pe.Node(interface=util.IdentityInterface(fields=['out_file']), name='outputs')

    within_run_align.connect(inputs, 'in_file', space_time_align, 'in_file')

    if slice_timing_correction:
        within_run_align.connect(inputs, 'in_file', get_meta, 'in_file')
        within_run_align.connect(get_meta, 'tr', space_time_align, 'tr')
        within_run_align.connect(get_meta, 'slice_times', select_slice_times, 'inlist')
        within_run_align.connect(select_slice_times, 'out', space_time_align, 'slice_times')
    
    within_run_align.connect(space_time_align, 'out_file', outputs, 'out_file')
    
    return within_run_align

#assumptions: align to first frame of first series
def create_between_run_align_workflow(name='between_run_align', ref_vol='first'):
    
    between_run_align = pe.Workflow(name='between_run_align')
    
    inputs = pe.Node(interface=util.IdentityInterface(fields=['in_files']), name='inputs')
    
    select_ref_vol = pe.Node(interface=util.Select(), name='select_ref_vol')
    select_ref_vol.inputs.index = [0]
    
    extract_ref_vol = pe.Node(interface=fsl.ExtractROI(), name='extract_ref_vol')
    extract_ref_vol.inputs.t_min = 0
    extract_ref_vol.inputs.t_size = 1
    
    motion_correction = pe.MapNode(interface=fsl.MCFLIRT(), name='motion_correction', iterfield=['in_file'])
    
    outputs = pe.Node(interface=util.IdentityInterface(fields=['out_files']), name='outputs')
    
    between_run_align.connect(inputs, 'in_files', select_ref_vol, 'inlist')
    between_run_align.connect(select_ref_vol, 'out', extract_ref_vol, 'in_file')
    between_run_align.connect(inputs, 'in_files', motion_correction, 'in_file')
    between_run_align.connect(extract_ref_vol, 'roi_file', motion_correction, 'ref_file')
    between_run_align.connect(motion_correction, 'out_file', outputs, 'out_files')
    
    return between_run_align

def create_preprocess_workflow(name,
                               work_dir,
                               sessions_file,
                               session_template,
                               scan_list,
                               fs_dir,
                               do_extract_inplane = True,
                               do_save_inplane = True,
                               do_align_to_anatomy = True,
                               do_align_qa = True,
                               do_save_align_qa = True,
                               do_save_strip = True,
                               do_save_align = True,
                               do_extract_functionals = True,
                               do_save_functionals = True,
                               do_within_run_align = True,
                               do_between_run_align = True,
                               do_merge_functionals = True,
                               do_save_merge = True):
    #initialize workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = work_dir

    ##for each session
    sessions_info = ColumnData(sessions_file, dtype=str)
    sessions = pe.Node(interface=util.IdentityInterface(fields=['session_dir','subject_id']), name='sessions')
    sessions.iterables = sessions_info.items()
    sessions.synchronize = True

    #get session directory
    get_session_dir = pe.Node(interface=nio.SelectFiles(session_template), name='get_session_dir')
    workflow.connect(sessions,'session_dir',get_session_dir,'session_dir')

    #save outputs
    datasink = pe.Node(nio.DataSink(), name='datasink')
    datasink.inputs.parameterization = False
    workflow.connect(get_session_dir,'session_dir',datasink,'base_directory')

    #extract inplane
    if do_extract_inplane:
        extract_inplane = create_extract_inplane_workflow()
        workflow.connect(get_session_dir,'session_dir',extract_inplane,'inputs.session_dir')

        if do_save_inplane:
            workflow.connect(extract_inplane,'outputs.out_file',datasink,'mri.@inplane')

            #align inplanes to anatomy
            if do_align_to_anatomy:
                get_anatomy = pe.Node(interface=nio.FreeSurferSource(), name='get_anatomy')
                get_anatomy.inputs.subjects_dir = fs_dir
                workflow.connect(sessions,'subject_id',get_anatomy,'subject_id')
                
                align_to_anatomy = create_align_to_anatomy_workflow()
                workflow.connect(extract_inplane,'outputs.out_file',align_to_anatomy,'inputs.inplane_file')
                workflow.connect(get_anatomy,'brain',align_to_anatomy,'inputs.anatomy_file')

                if do_align_qa:
                    align_qa = pe.Node(interface=nmutil.AlignmentQA(), name='align_qa')
                    workflow.connect(get_anatomy,'brain',align_qa,'target_file')
                    workflow.connect(align_to_anatomy,'outputs.strip_file',align_qa,'source_file')
                    workflow.connect(align_to_anatomy,'outputs.xfm_file',align_qa,'reg_file')

                    if do_save_align_qa:
                        workflow.connect(align_qa,'out_file',datasink,'qa.inplane_to_anatomy')

                if do_save_strip:
                    workflow.connect(align_to_anatomy,'outputs.strip_file',datasink,'mri.@inplane.@strip')

                if do_save_align:
                    workflow.connect(align_to_anatomy,'outputs.xfm_file',datasink,'mri.transforms.@inplane_to_anatomy')

    if do_extract_functionals:
        ##for each functional
        scans = pe.Node(interface=util.IdentityInterface(fields=['scan']), name='scans')
        scans.iterables = ('scan', scan_list)

        #extract functionals
        extract_functional = create_extract_functional_workflow()
        workflow.connect(get_session_dir,'session_dir',extract_functional,'inputs.session_dir')
        workflow.connect(scans,'scan',extract_functional,'inputs.scan')
        last_node = extract_functional

        #simultaneous slicing timing and motion correction
        if do_within_run_align:
            within_run_align = create_within_run_align_workflow()
            workflow.connect(last_node,'outputs.out_file',within_run_align,'inputs.in_file')
            last_node = within_run_align

        ##with all functionals
        join_functionals = pe.JoinNode(interface=util.IdentityInterface(fields=['functionals']),
                                       name='join_functionals',
                                       joinsource='scans')
        
        workflow.connect(last_node,'outputs.out_file',join_functionals,'functionals')

        #between run align
        if do_between_run_align:
            between_run_align = create_between_run_align_workflow()
            workflow.connect(join_functionals,'functionals',between_run_align,'inputs.in_files')

            workflow.connect(between_run_align,'outputs.out_files',datasink,'mri.@functionals')

            #merge functionals
            if do_merge_functionals:
                merge_functionals = pe.Node(interface=fsl.Merge(), name='merge_functionals')
                merge_functionals.inputs.dimension = 't'
                format_string = 'f'
                rename_merged = pe.Node(interface=util.Rename(format_string), name='rename_merged')
                rename_merged.inputs.keep_ext = True
                workflow.connect(between_run_align,'outputs.out_files',merge_functionals,'in_files')
                workflow.connect(merge_functionals,'merged_file',rename_merged,'in_file')

            if do_save_merge:
                workflow.connect(rename_merged,'out_file',datasink,'mri.@functionals.@merged')

    return workflow

