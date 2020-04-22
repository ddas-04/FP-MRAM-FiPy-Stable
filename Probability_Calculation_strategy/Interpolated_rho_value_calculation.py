#!/usr/bin/env python
########## Rho value read function from a vtk file ###########################
def get_rho_tuple(number_of_cells):
    Filename = './with_axis_0_img_00000.vtk' #This file contains the phi value
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(Filename)
    reader.ReadAllScalarsOn()
    reader.Update()
    usg = reader.GetOutput().GetCellData().GetScalars()
    z = []
    for i in range(number_of_cells):
        z.append(usg.GetTuple(i)[0])
    return tuple(z)


############# function to convert cartesian to spherical polar co-ordinate #########
def cart2pol(mValue):
    r=numerix.linalg.norm(mValue,axis=0)  
    theta=numerix.arccos(mValue[2,:]/r)
    phi=numerix.arctan2(mValue[1,:],mValue[0,:])
    # this section converts negative phi value to positive value by adding 2pi value
    neg_index=numerix.where(phi<0)
    phi[neg_index]=phi[neg_index]+2*np.pi
    
    return r,theta,phi

######## Index sorting function of y according to x #######################
def index_sorted(x,y):
    d = np.argsort(x)
    z = [0] * len(y)
    for i in range(len(d)):
        z[i] = y[d[i]]
    return z


############### Trapezoidal function for irregular grid size ###############
def trapz_irregular(x,y):
    n=len(x)
    index=1
    sum=0
    while index<=n-1:
        dx=x[index]-x[index-1]
        fx=0.5*(y[index-1]+y[index])
        sum=sum+fx*dx
        index=index+1
    
    return sum


################### Importing Modules ###############################
print('Import starts')

from fipy import FaceVariable, CellVariable, Gmsh2DIn3DSpace, VTKViewer, TransientTerm, ExplicitDiffusionTerm, DiffusionTerm, ExponentialConvectionTerm, DefaultSolver
from fipy.variables.variable import Variable
from fipy.tools import numerix
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
from shutil import copyfile 
import vtk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':20})

print('Import finished')

# ### Load mesh details from a saved file
mesh=pickle.load(open("./mesh_details_cellsize_0pt008_extrude_1pt00001.p","rb"))
gridCoor = mesh.cellCenters
mUnit = gridCoor
mNorm = numerix.linalg.norm(mUnit,axis=0)   
print('max mNorm='+str(max(mNorm))) 
print('min mNorm='+str(min(mNorm))) 
mAllCell = mUnit / mNorm

mAllCell_cartesian_save=mAllCell

#print(type(gridCoor))

msize=numerix.shape(mAllCell)
number_of_cells=msize[1]
print('Total number of cells = ' +str(number_of_cells))

################# Loading rho values from a saved file ##########################
rho_value = get_rho_tuple(number_of_cells) # get rho values from a previous .vtk file
rho = CellVariable(name=r"$\rho$",mesh=mesh,value=rho_value)

############# To convert into numpy array, save it in numpy file and again store it in a variable
np.save('rho_value.npy', rho)

rho_value=[]
rho_value=np.load('rho_value.npy')

# ### Convert m values into numpy array by saving and loading into a file
numerix.save('mValue_info.npy', mAllCell) 

# load the state space coordinate array values
mValue=numerix.load('mValue_info.npy')
np.shape(mValue)

# ### Convert m values from cartesian to spherical polar

mvalue_sph_pol=numerix.asarray(cart2pol(mValue))

Phi_deg = mvalue_sph_pol[2,:]*(180.0/np.pi)
print('Phi_max_degree = '+str(max(Phi_deg)))
print('Phi_min_degree = '+str(min(Phi_deg)))

phi_angle_all=mvalue_sph_pol[2,:]

phi_value=0.0*numerix.pi # initial phi value  
phi_step=1.0*360.0            # Phi value range(0,2pi) will be divided by phi_step
delta=(2*numerix.pi)/phi_step    # delta by which phi value will be increased
phi_save=[]  # this will save the values of Phi, which will be required to do the integration
total_size=0   # this will keep track the total number of cells being covered during calculation

m_cell_sph=[[],[],[]]

while phi_value< (2.0*numerix.pi):
    ######### Search index of phi where.....(phi_value-delta)<=phi_angle_all<(phi_value+delta) ##############
    index_phi=numerix.asarray(numerix.where((phi_angle_all>=phi_value) & (phi_angle_all<phi_value+delta)))
    #####################################################################################################
    r_at_index=mvalue_sph_pol[0,index_phi]
    theta_at_index=mvalue_sph_pol[1,index_phi]   # pulling out the values of the theta corresponding to index_phi
    
    ####################### Sorting rho according to sorted index of theta ########################
    xvar=theta_at_index[0,:] # storing the values of theta corresponding to index_phi into xvar
    
    rho_at_index=rho_value[index_phi] # pulling out the values of the rho corresponding to index_phi
    
    yvar=rho_at_index[0,:] # storing the values of rho corresponding to index_phi into xvar
        
    rho_sorted_acc_to_theta=index_sorted(xvar,yvar) # calling the index_sorted function 
    rho_sorted_acc_to_theta=np.reshape(rho_sorted_acc_to_theta,(1,len(rho_sorted_acc_to_theta))) # reshaping into a single array
    
    theta_sorted=np.sort(theta_at_index) # theta value sorted
    theta_sorted=theta_sorted[0,:] # converting from 1Xn matrix to row array of n elements 
    
    
    phi_new=phi_value*numerix.ones(len(theta_sorted))
    
    m_cell_modified_sph_pol=numerix.append(numerix.asarray(r_at_index),[numerix.asarray(theta_sorted),numerix.asarray(phi_new)],axis=0)
    
    #print(numerix.shape(m_cell_modified_sph_pol))
    m_cell_sph=numerix.append(m_cell_sph,(m_cell_modified_sph_pol),axis=1) #Appending the m values in spherical polar coordinate
    
    theta_sz=np.shape(theta_sorted)
    size=theta_sz[0]
    total_size=total_size+size # calculate total number of cells being covered during calculation
    
    phi_value=phi_value+delta # phi is increased by delta at end of each loop
    
print('Total number of cells covered = ' +str(total_size))
#type(m_cell_sph)

######## Converted the spherical polar coordinates back to cartesian coordinate #####################
m_x=(m_cell_sph[0,:]*numerix.sin(m_cell_sph[2,:])*numerix.cos(m_cell_sph[1,:]))
m_y=(m_cell_sph[0,:]*numerix.sin(m_cell_sph[2,:])*numerix.sin(m_cell_sph[1,:]))
m_z=m_cell_sph[0,:]*numerix.cos(m_cell_sph[2,:])

m_cellcenters_cart_modified=numerix.array([[m_x],[m_y],[m_z]])
#numerix.shape(m_cellcenters_cart_modified)


m_cellcenters_cart_modified=numerix.reshape(m_cellcenters_cart_modified,(3,total_size))

#mcell_new=m_cellcenters_cart_modified*CellVariable([1.0])

mcell_new=CellVariable(mesh=mesh, value=m_cellcenters_cart_modified)

#type(mcell_new)
#mcell_new.shape
#type(rho)
print('Calculation starts')

interpolated_rho=rho(mcell_new.globalValue, order=1)

pickle.dump(interpolated_rho, open("interpolated rho.p","wb") )
