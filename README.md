# FP-MRAM-FiPy-Stable
This is the stable version of the FiPy based Fokker-Planck solver.


First, the uniaxial field is applied only which created two poles along the uniaxial direction.


Second, an external magnetic field is added along with the uniaxial field, which forces the probability density to concentrate along the externally applied magnetic field(Condition is that, this external field magnitude should be much greater than the uniaxial field.


Then the total probability over the spherical surface is calculated, which comes to be ~1.


For calculating the WER, the probability code is modified to calculate at the upper/lower hemisphere. 


For the uniaxial field only, the probability at any hemisphere should be equal to ~0.5.


For externally applied magnetic field, the probability at the hemisphere, where the probability density lies should be ~1, whereas in the opposite hemisphere it should be ~0. 


For every calculation, meshing was done in a unit sphere for cellsize=0.08 and extrude=1.00001. It was not done by the usual FiPy method as it takes a huge time for meshing only. Instead, it was done by reading a .p file which contains the meshing information with the mentioned cellsize and extrude. As the file size is 311MB which exceeds the given size limit of Github for any single file, so it could not be uploaded on Github. You can get the file from the link below:

https://drive.google.com/file/d/1FA3WFNVujffLzOQ1NbtsojkCBS5XRHKy/view?usp=sharing

