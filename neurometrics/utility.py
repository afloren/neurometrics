from numpy import *
from parse import *
import scipy.io
import nibabel as nib
import h5py

_xfm_template = '''MNI Transform File
% {}

Transform_Type = Linear;
Linear_Transform =
{:>13.8f} {:>13.8f} {:>13.8f} {:>13.8f} 
{:>13.8f} {:>13.8f} {:>13.8f} {:>13.8f} 
{:>13.8f} {:>13.8f} {:>13.8f} {:>13.8f} ;
'''

_xfm_template2 = '''MNI Transform File
% {}

Transform_Type = Linear;
Linear_Transform =
{:>f} {:>f} {:>f} {:>f} 
{:>f} {:>f} {:>f} {:>f} 
{:>f} {:>f} {:>f} {:>f} ;
'''

_lta_template_header = '''type      = {type:d}
nxforms   = {nxforms:d}
mean      = {:>f} {:>f} {:>f}
sigma     = {sigma:>f}'''

_lta_template_xform = '''{d1:d} {d2:d} {d3:d}
{:>e} {:>e} {:>e} {:>e}
{:>e} {:>e} {:>e} {:>e}
{:>e} {:>e} {:>e} {:>e}
{:>e} {:>e} {:>e} {:>e}'''

_lta_template_src = '''src volume info        
valid = {:d}  # volume info valid
filename = {}
volume = {:d} {:d} {:d}
voxelsize = {:>f} {:>f} {:>f}
xras   = {:>f} {:>f} {:>f}
yras   = {:>f} {:>f} {:>f}
zras   = {:>f} {:>f} {:>f}
cras   = {:>f} {:>f} {:>f}'''

_lta_template_dst = '''dst volume info
valid = {:d}  # volume info valid
filename = {}
volume = {:d} {:d} {:d}
voxelsize = {:>f} {:>f} {:>f}
xras   = {:>f} {:>f} {:>f}
yras   = {:>f} {:>f} {:>f}
zras   = {:>f} {:>f} {:>f}
cras   = {:>f} {:>f} {:>f}'''

Axcodes = {'PIL' : tuple('PIL'),
           'PIR' : tuple('PIR'),
           'LPI' : tuple('LPI'),
           'PLI' : tuple('PLI'),
           'RAS' : tuple('RAS')}

Labels = {'RAS' : zip('LPI','RAS')}

def homogenize(rot=eye(3),trans=(0,0,0)):
    h = eye(4)
    h[:3,:3] = rot
    h[:3, 3] = trans
    return h

def load_lta(lta):
    with file(lta,'r') as f:
        lines = f.readlines()
        lines = [l.partition('#')[0].rstrip()+'\n' for l in lines] #strip comments
        lines = [l for l in lines if l != '\n'] #remove empty lines

        header = parse(_lta_template_header, ''.join(lines[:4]))

        if header['type'] != 1 or header['nxforms'] != 1:
            return None #don't know how to handle this case

        xform = parse(_lta_template_xform, ''.join(lines[4:9]))

        arr = array(xform.fixed).reshape((4,4))

        return arr

        

def load_xfm(xfm):
    res = parse(_xfm_template2,file(xfm,'r').read())
    arr = array(res.fixed[1:]).reshape((3,4))
    return arr

def save_xfm(arr, xfm):
    f = file(xfm,'w')
    f.write(_xfm_template.format('save_xfm',*arr.ravel()))
    f.close()

def bestrotvol2xfm(bestrotvol, xfm, 
                   input_axcodes=Axcodes['LPI'],
                   output_axcodes=Axcodes['PIL'],
                   xfm_axcodes=Axcodes['RAS']):

    mat = scipy.io.loadmat(bestrotvol)
    rot = mat['rot']
    trans = mat['trans']

    Mf = homogenize(rot=rot,trans=trans)

    A = homogenize(rot=ornt2rot(axcodes2ornt(xfm_axcodes,ref=output_axcodes)))
    B = homogenize(rot=ornt2rot(axcodes2ornt(input_axcodes,ref=xfm_axcodes)))

    #put transform in xfm_axcodes orientation
    Mf = A.dot(Mf).dot(B)

    f = file(xfm,'w')
    f.write(_xfm_template.format('bestrotvol2xfm',*Mf.ravel()))
    f.close()

def ornt2rot(ornt):
    ornt = asarray(ornt)
    p = ornt.shape[0]
    reorder = eye(p)[:, ornt[:,0]]
    flip = diag(ornt[:,1])
    return reorder.dot(flip)

def axcodes2ornt(axcodes,ref=Axcodes['RAS'],labels=Labels['RAS']):
    n_axes = len(axcodes)
    ornt = zeros((n_axes, 2), dtype=int8)
    
    for i, a in enumerate(axcodes):
        for j, r in enumerate(ref):
            for k, l in enumerate(labels):
                if a in l and r in l:
                    if a == r:
                        ornt[j, :] = [i, 1]
                    else:
                        ornt[j, :] = [i,-1]

    #TODO: error checking
    return ornt

def arr2nifti(arr,scale=(1.0,1.0,1.0),axcodes=Axcodes['RAS']):
    ornt = axcodes2ornt(axcodes)#TODO: update version of nibabel to use builtin axcodes2ornt
    rot = ornt2rot(ornt).dot(diag(scale))
    aff = homogenize(rot=rot)#TODO: need to fixup translation of center
    nim = nib.nifti1.Nifti1Image(arr,aff)
    return nim

def recursive_hstack(arr):
    arr = hstack(arr)
    return recursive_hstack(arr) if arr.dtype == object else arr

def vista2nifti(vista_file):
    vista = scipy.io.loadmat(vista_file)
    arr = vista['anat']
    scale = recursive_hstack(vista['inplanes']['voxelSize']).ravel()
    return arr2nifti(arr,scale=scale,axcodes=Axcodes['PLI'])

#import subprocess

#def mri_info(args):
#    return subprocess.check_output(['mri_info']+args)

#from StringIO import StringIO

def get_vox2ras(file_name):
    return nib.load(file_name).get_affine()
    #return loadtxt(StringIO(mri_info(['--vox2ras',file_name])))

def get_shape(file_name):
    return nib.load(file_name).shape
    #nrows = int(mri_info(['--nrows',file_name]).strip())
    #ncols = int(mri_info(['--ncols',file_name]).strip())
    #nslices = int(mri_info(['--nslices',file_name]).strip())
    #return (nrows,ncols,nslices)

def recenter_xfm(move,targ,xfm):
    #calculate transform to recenter mov such that the voxel 
    #space origins of mov and targ are aligned in RAS space
    #use mri_vol2vol --mov mov --targ targ --xfm xfm --o xfmdvol.nii.gz
    #to actually resample mov into this space

    move_vox2ras = get_vox2ras(move)
    move_shape = get_shape(move)
    move_axcodes = nib.orientations.aff2axcodes(move_vox2ras)

    targ_vox2ras = get_vox2ras(targ)
    targ_shape = get_shape(targ)
    targ_axcodes = nib.orientations.aff2axcodes(targ_vox2ras)

    #TODO: test code for various axcode configurations
    ornt = axcodes2ornt(move_axcodes,targ_axcodes)
    aff = nib.orientations.orientation_affine(ornt,targ_shape)
    c = targ_vox2ras.dot(aff)[:3,3]

    transform = homogenize(trans = c)

    save_xfm(transform, xfm)
    #f = file(xfm,'w')
    #f.write(_xfm_template.format(*transform.ravel()))
    #f.close()
    
        

    
    
