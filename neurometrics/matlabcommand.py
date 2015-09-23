from os import path
from tempfile import mkdtemp
from shutil import rmtree
from scipy.io import savemat, loadmat
from subprocess import check_call, PIPE

template = """
try
  load input.mat;
  {0};
  save output.mat;
catch err
  disp(err.getReport());
  quit (1);
end

quit(0);
"""

def matlab_command(cmd, matlab='matlab', **kwargs):
    dir = mkdtemp()
    try:
        ifile = path.join(dir,'input.mat')
        mfile = path.join(dir,'script.m')
        ofile = path.join(dir,'output.mat')

        savemat(ifile,kwargs)

        m = open(mfile,'w')
        m.write(template.format(cmd))
        m.close()

        check_call([matlab,'-nodesktop','-nosplash','-r','script'],cwd=dir)

        output = loadmat(ofile)
    finally:
        rmtree(dir)

    return output

