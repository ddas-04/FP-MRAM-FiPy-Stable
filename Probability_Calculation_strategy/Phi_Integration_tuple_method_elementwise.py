def get_rho_tuple(number_of_cells):
    #Filename = '/home/debasis/MEGA/My_Github_repositories/FP_MRAM_FiPy_stable/Fresh_start/90_degree/Uniaxial_field_only_uniform_rho/with_axis_0_VTK_files/with_axis_0_img_00000.vtk' #This file contains the phi value
    Filename = '/home/eledd/FP_work/Fipy_code/Fresh_start/90_degree/Uniaxial_only_from_uniform_rho/with_axis_0_VTK_files/with_axis_0_img_00401.vtk'
    #Filename = '/home/debasis/MEGA/My_Github_repositories/FP_MRAM_FiPy_stable/Fresh_start/90_degree/Pavans_trick/with_axis_0_VTK_files/with_axis_0_img_00401.vtk'
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(Filename)
    reader.ReadAllScalarsOn()
    reader.Update()
    usg = reader.GetOutput().GetCellData().GetScalars()
    z = []
    for i in range(number_of_cells):
        z.append(usg.GetTuple(i)[0])
    return tuple(z)

##### Internal functions
def getCellVariableDatapoint(coord):
  # expect coord to be a nested list
  #   coord[0][0] = x-coordinate
  #   coord[1][0] = y-coordinate
  #   coord[2][0] = z-coordinate
  global rho
  return rho(coord, order=1)

def sphericalDatapoint(theta, phi):
  # Expect theta and rho to be in radians
  #   theta is the angle from the +z-axis to the coordinate vector
  #   rho is the angle from the +x-axis to projection of the coordinate
  #     vector onto the xy-plane
  sin_theta = numerix.sin(theta)
  cos_theta = numerix.cos(theta)
  sin_phi = numerix.sin(phi)
  cos_phi = numerix.cos(phi)
  return sin_theta*getCellVariableDatapoint([[sin_theta*cos_phi], [sin_theta*sin_phi], [cos_theta]])


# ### Importing Modules
print('Import starts')
from fipy import FaceVariable, CellVariable, Gmsh2DIn3DSpace, VTKViewer, TransientTerm, ExplicitDiffusionTerm, DiffusionTerm, ExponentialConvectionTerm, DefaultSolver #, getCellVariableDatapoint

from fipy.variables.variable import Variable
from fipy.tools import numerix
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
from shutil import copyfile 
import vtk
import numpy as np
import matplotlib
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':20})
print('Import finished')

# ### Load mesh details from a saved file

mesh=pickle.load(open("/home/eledd/FP_work/Fipy_code/mesh_details_cellsize_0pt008_extrude_1pt00001.p","rb"))
gridCoor = mesh.cellCenters
mUnit = gridCoor
mNorm = numerix.linalg.norm(mUnit,axis=0)   
print('max mNorm='+str(max(mNorm))) 
print('min mNorm='+str(min(mNorm))) 
mAllCell = mUnit / mNorm

msize=numerix.shape(mAllCell)
number_of_cells=msize[1]
print('Total number of cells = ' +str(number_of_cells))

# ### Loading phi values from a saved file

rho_value = get_rho_tuple(number_of_cells) # get phi values from a previous .vtk file
rho = CellVariable(name=r"$\Rho$",mesh=mesh,value=rho_value)


##########################################################################
#                                    Check from here                                                                                 #
##########################################################################

nth=91
nphi=181
theta=numerix.linspace(0,numerix.pi,nth)
phi=numerix.linspace(0,2*numerix.pi,nphi)

probability=np.zeros(nth*nphi)
conv_fac=(180.0/numerix.pi)
file = open("Probability_output_element.txt","w")

f = lambda y, x: sphericalDatapoint(y, x)
i=0
for iphi in range(len(phi)-1):
	for ith in range(len(theta)-1):
		print('----------------------------------------------------------')
		print(i)
		print('Phi = ' +str(phi[iphi]*conv_fac) + ' to ' + str(phi[iphi+1]*conv_fac))
		print('Theta = ' + str(theta[ith]*conv_fac) + ' to ' + str(theta[ith+1]*conv_fac))
		[probability[i],error] = (dblquad(f, phi[iphi], phi[iphi+1], lambda x:  theta[ith], lambda x: theta[ith+1], epsabs=1e-13, epsrel=1e-5))
		print('Probability = ' + str(probability[i]))
		file.write('\n Segment ' + str(i) + 'Probability = ' + str(probability[i]))
		i=i+1
file.write('\n-----------------------------------------------------------------')
total_probability = numerix.sum(probability)


#[probability,err]=(dblquad(f, 0, numerix.pi/6.0, lambda x:  0, lambda x: 1.0*numerix.pi, epsabs=1e-13, epsrel=1e-5))
print('Probability = ' + str(total_probability))

file.write('Total Probability = ' + str(total_probability))
file.close()
