# FP-MRAM-FiPy-Stable
This is the stable version of the FiPy based Fokker-Planck solver.


First, the uniaxial field is applied only which created two poles along the uniaxial direction.


Second, an external magnetic field is added along with the uniaxial field, which forces the probility density to concentrate along the externally applied magnetic field(Condition is that, this external field magnitude should be much greater than the uniaxial field.


Then the total probability over the spherical surface is calculated, which comes to be ~1.


For calculating the WER, the probability code is modified to calculate at upper/lower hemisphere. 


For uniaxial field only, the probability at any hemisphere should be equal to ~0.5


For externally applied magnetic field, the probability at the hemisphere, where the probability density lies should be ~1, whereas in the opposite hemisphere it should be ~0. 

