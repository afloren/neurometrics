import os

import nipype.interfaces.io as nio
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.dcmstack as ds
import nipype.interfaces.nipy as nipy
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe

import neurometrics.interfaces.utils as nmutil

from mvpa2.misc.io import ColumnData

def ref_vol_to_inplane_id(ref_vol):
    return '1' if int(ref_vol) == -1 else ''

def create_extract_inplane_workflow(name='extract_inplane',
                                    templates={'inplane':'Raw/Anatomy/Inplane{id}/'},
                                    format_string='inplane'):

    extract_inplane = pe.Workflow(name=name)

    inputs = pe.Node(interface=util.IdentityInterface(fields=['session_dir','ref_vol']), name='inputs')
    get_inplane = pe.Node(interface=nio.SelectFiles(templates), name='get_inplane')
    inplane_to_nii = pe.Node(interface=ds.DcmStack(), name='inplane_to_nii')
    inplane_to_nii.inputs.embed_meta = True
    rename_inplane = pe.Node(interface=util.Rename(format_string), name='rename_inplane')
    rename_inplane.inputs.keep_ext = True
    outputs = pe.Node(interface=util.IdentityInterface(fields=['out_file']), name='outputs')
    
    extract_inplane.connect(inputs,'session_dir',get_inplane,'base_directory')
    extract_inplane.connect(inputs,('ref_vol', ref_vol_to_inplane_id),get_inplane,'id')
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
    #register.inputs.init_orient = True #FIXME: disabled due to bug in binary
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

        select_slice_times = pe.Node(interface=util.Select(), name='select_slice_times')#FIXME: sometimes required depending on dicom
        select_slice_times.inputs.index = [0]

    space_time_align = pe.Node(interface=nipy.SpaceTimeRealigner(), name='space_time_align')

    if slice_timing_correction:
        space_time_align.inputs.slice_info = 2

    outputs = pe.Node(interface=util.IdentityInterface(fields=['out_file']), name='outputs')

    within_run_align.connect(inputs, 'in_file', space_time_align, 'in_file')

    if slice_timing_correction:
        within_run_align.connect(inputs, 'in_file', get_meta, 'in_file')
        within_run_align.connect(get_meta, 'tr', space_time_align, 'tr')
        within_run_align.connect(get_meta, 'slice_times', select_slice_times, 'inlist')#see above
        within_run_align.connect(select_slice_times, 'out', space_time_align, 'slice_times')
        #within_run_align.connect(get_meta, 'slice_times', space_time_align, 'slice_times')
    
    within_run_align.connect(space_time_align, 'out_file', outputs, 'out_file')
    
    return within_run_align

def to_list(val):
    return [int(val)]

def to_int(val):
    return int(val)

#assumptions: align to first frame of first series
def create_between_run_align_workflow(name='between_run_align'):
    
    between_run_align = pe.Workflow(name=name)
    
    inputs = pe.Node(interface=util.IdentityInterface(fields=['in_files', 'ref_vol']), name='inputs')
    
    select_ref_vol = pe.Node(interface=util.Select(), name='select_ref_vol')
    #select_ref_vol.inputs.index = [ref_vol]

    extract_ref_vol = pe.Node(interface=nmutil.ExtractVolume(), name='extract_ref_vol')
    
    #extract_ref_vol = pe.Node(interface=fsl.ExtractROI(), name='extract_ref_vol')
    #extract_ref_vol.inputs.t_min = 0
    #extract_ref_vol.inputs.t_size = 1
    
    motion_correction = pe.MapNode(interface=fsl.MCFLIRT(), name='motion_correction', iterfield=['in_file'])
    
    outputs = pe.Node(interface=util.IdentityInterface(fields=['out_files']), name='outputs')
    
    between_run_align.connect(inputs, 'in_files', select_ref_vol, 'inlist')
    between_run_align.connect(inputs, ('ref_vol', to_list), select_ref_vol, 'index')
    between_run_align.connect(select_ref_vol, 'out', extract_ref_vol, 'in_file')
    between_run_align.connect(inputs, ('ref_vol', to_int), extract_ref_vol, 'index')
    between_run_align.connect(inputs, 'in_files', motion_correction, 'in_file')
    between_run_align.connect(extract_ref_vol, 'out_file', motion_correction, 'ref_file')
    between_run_align.connect(motion_correction, 'out_file', outputs, 'out_files')
    
    return between_run_align

def create_ml_vol_workflow(name='ml_vol',
                           do_summarize=True): 
    workflow = pe.Workflow(name=name)

    nifti_to_ds = pe.Node(nmutil.NiftiToDataset(), name='nifti_to_ds')    
    
    ml = pe.Node(nmutil.PerformML(), name='ml')
    workflow.connect(nifti_to_ds,'ds_file',ml,'ds_file')

    if do_summarize:
        summarize = pe.Node(nmutil.SummarizeResults(), name='summarize')
        workflow.connect(ml,'results_file',summarize,'results_files')
    
    return workflow

def create_ml_surface_workflow(name='ml_surface',
                               do_summarize=True):
    workflow = pe.Workflow(name=name)

    nifti_to_ds = pe.MapNode(nmutil.NiftiToDataset(), name='nifti_to_ds', iterfield='nifti_file')

    join_hemispheres = pe.Node(nmutil.JoinDatasets(), name='join_hemispheres')
    join_hemispheres.inputs.join_hemispheres = True
    workflow.connect(nifti_to_ds,'ds_file',join_hemispheres,'input_datasets')
    
    ml = pe.Node(nmutil.PerformML(), name='ml')
    workflow.connect(join_hemispheres,'joined_dataset',ml,'ds_file')

    if do_summarize:
        summarize = pe.Node(nmutil.SummarizeResults(), name='summarize')
        workflow.connect(ml,'results_file',summarize,'results_files')
    
    return workflow

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
                               do_slice_timing_correction = True,
                               do_between_run_align = True,
                               do_merge_functionals = True,
                               do_within_subject_align = True,
                               do_save_merge = True):
    #initialize workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = work_dir

    ##for each session
    sessions_info = ColumnData(sessions_file, dtype=str)
    sessions = pe.Node(interface=util.IdentityInterface(fields=['session_dir','subject_id','ref_vol']), name='sessions')
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
        workflow.connect(sessions,'ref_vol',extract_inplane,'inputs.ref_vol')

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
            within_run_align = create_within_run_align_workflow(slice_timing_correction = do_slice_timing_correction)
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
            workflow.connect(sessions,'ref_vol',between_run_align,'inputs.ref_vol')

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

def create_ml_preprocess_workflow(name,
                                  project_dir,
                                  work_dir,
                                  sessions_file,
                                  session_template,
                                  fs_dir,
                                  fwhm_vals=[2],
                                  ico_order_vals=[4],
                                  do_save_vol_ds = False,
                                  do_save_smooth_vol_ds = False,
                                  do_save_surface_smooth_vol_ds = False,
                                  do_save_surface_ds = False,
                                  do_save_smooth_surface_ds = False,
                                  do_save_sphere_nifti = False,
                                  do_save_sphere_ds = True,
                                  do_save_join_sessions_ds = True,
                                  do_save_join_subjects_ds = True):

    #initialize workflow                                                                                   
    workflow = pe.Workflow(name=name)
    workflow.base_dir = work_dir

    sessions_info = ColumnData(sessions_file, dtype=str)
    subject_ids = set(sessions_info['subject_id'])
    session_map = [(sid,[s for i,s,r in zip(*sessions_info.values()) if i == sid])
                   for sid in subject_ids]

    ##for each subject                                                                                         
    subjects = pe.Node(interface=util.IdentityInterface(fields=['subject_id']), name='subjects')
    subjects.iterables = [('subject_id', subject_ids)]

    ##for each session                                                                         
    sessions = pe.Node(interface=util.IdentityInterface(fields=['subject_id','session_dir']), name='sessions')
    sessions.itersource = ('subjects','subject_id')
    sessions.iterables = [('session_dir', dict(session_map))]
    workflow.connect(subjects,'subject_id',sessions,'subject_id')

    #get session directory                                                                                                        
    get_session_dir = pe.Node(interface=nio.SelectFiles(session_template), name='get_session_dir')
    workflow.connect(sessions,'session_dir',get_session_dir,'session_dir')

    #save outputs
    datasink = pe.Node(nio.DataSink(), name='datasink')
    datasink.inputs.parameterization = False
    workflow.connect(get_session_dir,'session_dir',datasink,'base_directory')

    template = {'nifti_file':'mri/f.nii.gz',
                'attributes_file':'attributes.txt',
                'reg_file':'mri/transforms/functional_to_anatomy.dat'}
    get_files = pe.Node(nio.SelectFiles(template), name='get_files')
    workflow.connect(get_session_dir,'session_dir',get_files,'base_directory')

    vol_to_ds = pe.Node(nmutil.NiftiToDataset(), name='vol_to_ds')
    vol_to_ds.inputs.ds_file = 'vol.hdf5'

    workflow.connect(get_files,'nifti_file',vol_to_ds,'nifti_file')
    workflow.connect(get_files,'attributes_file',vol_to_ds,'attributes_file')
    workflow.connect(subjects,'subject_id',vol_to_ds,'subject_id')
    workflow.connect(sessions,'session_dir',vol_to_ds,'session_id')

    if do_save_vol_ds:
        workflow.connect(vol_to_ds,'ds_file',datasink,'ml.@vol')

    fwhm = pe.Node(util.IdentityInterface(fields=['fwhm']), name='fwhm')
    fwhm.iterables = [('fwhm',fwhm_vals)]

    if do_save_smooth_vol_ds:
        smooth_vol = pe.Node(interface=fs.MRIConvert(), name='smooth_vol')
        workflow.connect(get_files,'nifti_file',smooth_vol,'in_file')
        workflow.connect(fwhm,'fwhm',smooth_vol,'fwhm')
    
        smooth_vol_to_ds = pe.Node(nmutil.NiftiToDataset(), name='smooth_vol_to_ds')
        smooth_vol_to_ds.inputs.ds_file = 'smooth_vol.hdf5'
    
        workflow.connect(smooth_vol,'out_file',smooth_vol_to_ds,'nifti_file')
        workflow.connect(get_files,'attributes_file',smooth_vol_to_ds,'attributes_file')
        workflow.connect(subjects,'subject_id',smooth_vol_to_ds,'subject_id')
        workflow.connect(sessions,'session_dir',smooth_vol_to_ds,'session_id')
    
        workflow.connect(smooth_vol_to_ds,'ds_file',datasink,'ml.@smooth_vol')

    if do_save_surface_smooth_vol_ds:
        surface_smooth_vol = pe.Node(interface=fs.Smooth(), name='surface_smooth_vol')
        workflow.connect(get_files,'reg_file',surface_smooth_vol,'reg_file')
        workflow.connect(get_files,'nifti_file',surface_smooth_vol,'in_file')
        workflow.connect(fwhm,'fwhm',surface_smooth_vol,'surface_fwhm')
    
        surface_smooth_vol_to_ds = pe.Node(nmutil.NiftiToDataset(), name='surface_smooth_vol_to_ds')
        surface_smooth_vol_to_ds.inputs.ds_file = 'surface_smooth_vol.hdf5'
    
        workflow.connect(surface_smooth_vol,'out_file',surface_smooth_vol_to_ds,'nifti_file')
        workflow.connect(get_files,'attributes_file',surface_smooth_vol_to_ds,'attributes_file')
        workflow.connect(subjects,'subject_id',surface_smooth_vol_to_ds,'subject_id')
        workflow.connect(sessions,'session_dir',surface_smooth_vol_to_ds,'session_id')
    
        workflow.connect(surface_smooth_vol_to_ds,'ds_file',datasink,'ml.@surface_smooth_vol')

    hemi = pe.Node(util.IdentityInterface(fields=['hemi']), name='hemi')
    hemi.iterables = [('hemi',['lh','rh'])]

    to_surface = pe.Node(fs.SampleToSurface(), name='to_surface')
    to_surface.inputs.sampling_method = 'average'
    to_surface.inputs.sampling_range = (0., 1., 0.1)
    to_surface.inputs.sampling_units = 'frac'
    to_surface.inputs.subjects_dir = fs_dir
    workflow.connect(hemi,'hemi',to_surface,'hemi')
    workflow.connect(get_files,'nifti_file',to_surface,'source_file')
    workflow.connect(get_files,'reg_file',to_surface,'reg_file')

    if do_save_surface_ds:    
        surface_to_ds = pe.Node(nmutil.NiftiToDataset(), name='surface_to_ds')
        workflow.connect(to_surface,'out_file',surface_to_ds,'nifti_file')
        workflow.connect(get_files,'attributes_file',surface_to_ds,'attributes_file')
        workflow.connect(subjects,'subject_id',surface_to_ds,'subject_id')
        workflow.connect(sessions,'session_dir',surface_to_ds,'session_id')

        join_surfaces = pe.JoinNode(nmutil.JoinDatasets(), 
                                    name='join_surfaces',
                                    joinsource='hemi',
                                    joinfield='input_datasets')
        join_surfaces.inputs.joined_dataset = 'surface.hdf5'
        join_surfaces.inputs.join_hemispheres = True
        workflow.connect(surface_to_ds,'ds_file',join_surfaces,'input_datasets')
    
        workflow.connect(join_surfaces,'joined_dataset',datasink,'ml.@surface')

    smooth_surface = pe.Node(fs.SurfaceSmooth(), name='smooth_surface')
    smooth_surface.inputs.subjects_dir = fs_dir
    workflow.connect(to_surface,'out_file',smooth_surface,'in_file')
    workflow.connect(sessions,'subject_id',smooth_surface,'subject_id')
    workflow.connect(hemi,'hemi',smooth_surface,'hemi')
    workflow.connect(fwhm,'fwhm',smooth_surface,'fwhm')

    if do_save_smooth_surface_ds:        
        smooth_surface_to_ds = pe.Node(nmutil.NiftiToDataset(), name='smooth_surface_to_ds')
        workflow.connect(smooth_surface,'out_file',smooth_surface_to_ds,'nifti_file')
        workflow.connect(get_files,'attributes_file',smooth_surface_to_ds,'attributes_file')
        workflow.connect(subjects,'subject_id',smooth_surface_to_ds,'subject_id')
        workflow.connect(sessions,'session_dir',smooth_surface_to_ds,'session_id')

        join_smooth_surfaces = pe.JoinNode(nmutil.JoinDatasets(), 
                                           name='join_smooth_surfaces',
                                           joinsource='hemi',
                                           joinfield='input_datasets')
        join_smooth_surfaces.inputs.joined_dataset = 'smooth_surface.hdf5'
        join_smooth_surfaces.inputs.join_hemispheres = True
        workflow.connect(smooth_surface_to_ds,'ds_file',join_smooth_surfaces,'input_datasets')
    
        workflow.connect(join_smooth_surfaces,'joined_dataset',datasink,'ml.@smooth_surface')
    

    ico_order = pe.Node(util.IdentityInterface(fields=['ico_order']), name='ico_order')
    ico_order.iterables = [('ico_order',ico_order_vals)]

    to_sphere = pe.Node(fs.SurfaceTransform(), name='to_sphere')
    to_sphere.inputs.target_subject = 'ico'
    to_sphere.inputs.subjects_dir = fs_dir
    workflow.connect(hemi,'hemi',to_sphere,'hemi')
    workflow.connect(smooth_surface,'out_file',to_sphere,'source_file')
    workflow.connect(subjects,'subject_id',to_sphere,'source_subject')
    workflow.connect(ico_order,'ico_order',to_sphere,'target_ico_order')

    if do_save_sphere_nifti:
        workflow.connect(to_sphere,'out_file',datasink,'surf.@sphere')

    template = {'annot_file':'{subject_id}/label/{hemi}.aparc.a2009s.annot'}
    get_annot_file = pe.Node(nio.SelectFiles(template), name='get_annot_file')
    get_annot_file.inputs.base_directory = fs_dir
    get_annot_file.inputs.subject_id = 'fsaverage'
    workflow.connect(hemi,'hemi',get_annot_file,'hemi')

    transform_annot = pe.Node(fs.SurfaceTransform(), name='transform_annot')
    transform_annot.inputs.source_subject = 'fsaverage'
    transform_annot.inputs.target_subject = 'ico'
    transform_annot.inputs.subjects_dir = fs_dir
    workflow.connect(hemi,'hemi',transform_annot,'hemi')
    workflow.connect(get_annot_file,'annot_file',transform_annot,'source_annot_file')
    workflow.connect(ico_order,'ico_order',transform_annot,'target_ico_order')
    
    sphere_to_ds = pe.Node(nmutil.NiftiToDataset(), name='sphere_to_ds')
    workflow.connect(to_sphere,'out_file',sphere_to_ds,'nifti_file')
    workflow.connect(get_files,'attributes_file',sphere_to_ds,'attributes_file')
    workflow.connect(transform_annot,'out_file',sphere_to_ds,'annot_file')
    workflow.connect(subjects,'subject_id',sphere_to_ds,'subject_id')
    workflow.connect(sessions,'session_dir',sphere_to_ds,'session_id')

    join_hemispheres = pe.JoinNode(nmutil.JoinDatasets(), 
                                   name='join_hemispheres',
                                   joinsource='hemi',
                                   joinfield='input_datasets')
    join_hemispheres.inputs.joined_dataset = 'sphere.hdf5'
    join_hemispheres.inputs.join_hemispheres = True

    workflow.connect(sphere_to_ds,'ds_file',join_hemispheres,'input_datasets')

    if do_save_sphere_ds:
        workflow.connect(join_hemispheres,'joined_dataset',datasink,'ml.@sphere')

    join_sessions = pe.JoinNode(nmutil.JoinDatasets(), 
                                name='join_sessions',
                                joinsource='sessions',
                                joinfield='input_datasets')
    workflow.connect(join_hemispheres,'joined_dataset',join_sessions,'input_datasets')

    if do_save_join_sessions_ds:
        join_sessions_sink = pe.Node(nio.DataSink(), name='join_sessions_sink')
        join_sessions_sink.inputs.parameterization = False
        join_sessions_sink.inputs.base_directory = os.path.join(project_dir,'ml')
        workflow.connect(subjects,'subject_id',join_sessions_sink,'container')
        workflow.connect(join_sessions,'joined_dataset',join_sessions_sink,'@join_sessions')

    join_subjects = pe.JoinNode(nmutil.JoinDatasets(),
                                name='join_subjects',
                                joinsource='subjects',
                                joinfield='input_datasets')
    workflow.connect(join_sessions,'joined_dataset',join_subjects,'input_datasets')

    if do_save_join_subjects_ds:
        join_subjects_sink = pe.Node(nio.DataSink(), name='join_subjects_sink')
        join_subjects_sink.inputs.parameterization = False
        join_subjects_sink.inputs.base_directory = os.path.join(project_dir,'ml')
        workflow.connect(join_subjects,'joined_dataset',join_subjects_sink,'@join_subjects')

    return workflow

def create_within_subject_workflow(name,
                                   work_dir,
                                   sessions_file,
                                   session_template,
                                   scan_list,
                                   fs_dir):
    
    #initialize workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = work_dir

    sessions_info = ColumnData(sessions_file, dtype=str)
    subject_ids = set(sessions_info['subject_id'])
    session_map = [(sid,[s for i,s in zip(*sessions_info.values()) if i == sid])
                   for sid in subject_ids]

    ##for each subject
    subjects = pe.Node(interface=util.IdentityInterface(fields=['subject_id']), name='subjects')
    subjects.iterables = [('subject_id', subject_ids)]

    ##for each session
    sessions = pe.Node(interface=util.IdentityInterface(fields=['subject_id','session_dir']), name='sessions')
    sessions.itersource = ('subjects','subject_id')
    sessions.iterables = [('session_dir', dict(session_map))]
    workflow.connect(subjects,'subject_id',sessions,'subject_id')

    #get session directory
    get_session_dir = pe.Node(interface=nio.SelectFiles(session_template), name='get_session_dir')
    workflow.connect(sessions,'session_dir',get_session_dir,'session_dir')

    #save outputs
    datasink = pe.Node(nio.DataSink(), name='datasink')
    datasink.inputs.parameterization = False
    workflow.connect(get_session_dir,'session_dir',datasink,'base_directory')

    template = {'functional':'mri/f.nii.gz'}
    get_files = pe.Node(nio.SelectFiles(template), name='get_files')
    workflow.connect(get_session_dir,'session_dir',get_files,'base_directory')

    join_sessions = pe.JoinNode(interface=util.IdentityInterface(fields=['functionals']),
                                name='join_sessions',
                                joinsource='sessions')
    workflow.connect(get_files,'functional',join_sessions,'functionals')

    within_subject_align = create_between_run_align_workflow(name='within_subject_align')
    workflow.connect(join_sessions,'functionals',within_subject_align,'inputs.in_files')
    
    return workflow
