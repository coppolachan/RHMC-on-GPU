/*! 
  Initialization routines
 */

#include <iostream>
#include "include/init.h"
#include "include/parameters.h"
#include "include/global_var.h"
#include "include/global_const.h"
#include "include/tools.h" //for random number generators
#include "include/geometry.h"
#include "lib/Tools/exception.h"

int initialize(void)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside init ..."<<endl;
  #endif
  
  try{
     // initialize random number generator
     initrand(rand_seed);

     // allocate and initialize rational approximations
     rationalapprox_calc();

     // staggered phases
     eta = new int[no_links];

     // initialize geometry
     // defined in Geometrty/geometry.cc
     init_geo();

     // allocate gauge configuration
     gauge_conf=new Conf(start);

     // allocate staples 
     gauge_staples=new Staples();

     // allocate momenta
     gauge_momenta=new Momenta();
     #ifdef REVERSIBILITY_TEST
       gauge_momenta_save=new Momenta();
     #endif

     // allocate ipdot
     gauge_ipdot=new Ipdot();

     // auxiliary vectors for global sums
     d_vector1=new double[size];
     d_vector2=new double[size];
     c_vector1=new complex<double>[size]; 
     c_vector2=new complex<double>[size];
     plaq_test=new double[size];

     // allocate fermions
     fermion_phi=new MultiFermion;
     fermion_chi=new MultiFermion;

     // allocate auxiliary global fermion
     loc_r=new Fermion;
     loc_h=new Fermion;
     loc_s=new Fermion;
     loc_p=new Fermion;

     // allocate shift fermions
     p_shiftferm=new ShiftFermion;
     fermion_shiftmulti=new ShiftMultiFermion;

     // allocate fermions for meas_chiral
     chi_e=new Fermion;
     chi_o=new Fermion;
     phi_e=new Fermion;
     phi_o=new Fermion;
     rnd_e=new Fermion;
     rnd_o=new Fermion;

     // to be used in cuda_rhmc
     #ifdef USE_GPU
     simple_fermion_packed = new float[6*sizeh*2];             
     // 6*sizeh = fermion,  2 since 1double~2float
     
     chi_packed            = new float[6*sizeh*no_ps*2];
     // 6*sizeh = fermion,  2 since 1double~2float
     
     psi_packed            = new float[6*sizeh*max_approx_order*no_ps*2]; 
     // 2 since 1double~2float
     
     gauge_field_packed    = new float[12*no_links*2];            
     // 2 since 1double~2float
     
     ipdot_packed          = new float[8*no_links];
     
     momenta_packed        = new float[8*no_links];
     
     shift_table           = new int[8*size];
     #endif

     // test if the simulation details are applicable
     test_param(); 
     }
  catch (exception& e)
     {
     cout << e.what() << endl;
     #ifdef DEBUG_MODE
     cout << "\tterminated init with exceptions"<< endl;
     #endif
     return 1;
     }

  #ifdef DEBUG_MODE
  cout << "\tterminated init without exceptions"<< endl;
  #endif

  return 0;
  }



void finalize(void)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside end ..."<< endl;
  #endif

  // clear rational approximations
  delete first_inv_approx_norm_coeff;
  delete md_inv_approx_norm_coeff;
  delete last_inv_approx_norm_coeff;
  delete meas_inv_coeff;

  // clear fermions for meas_chiral
  delete chi_e;
  delete chi_o;
  delete phi_e;
  delete phi_o;
  delete rnd_e;
  delete rnd_o;

  // clear shifted fermions
  delete p_shiftferm;
  delete fermion_shiftmulti;

  // clear auxiliary global fermions
  delete loc_r;
  delete loc_h;
  delete loc_s;
  delete loc_p;

  // clear fermions
  delete fermion_phi;
  delete fermion_chi;

  // clear auxiliary vectors
  delete [] plaq_test;
  delete [] d_vector1;
  delete [] d_vector2;
  delete [] c_vector1;
  delete [] c_vector2;

  // clear ipdot
  delete gauge_ipdot;

  // clear momenta
  delete gauge_momenta;  
  #ifdef REVERSIBILITY_TEST
    delete gauge_momenta_save;
  #endif

  // clear staples
  delete gauge_staples;

  // clear gauge configuration
  delete gauge_conf;

  // clear geometry variables
  end_geo();

  // clear staggered phases
  delete [] eta;

  #ifdef USE_GPU
  delete [] simple_fermion_packed;
  delete [] chi_packed;
  delete [] psi_packed;
  delete [] gauge_field_packed;
  delete [] ipdot_packed;
  delete [] momenta_packed;
  delete [] shift_table;
  #endif

  #ifdef DEBUG_MODE
  cout << "\tterminated end"<< endl;
  #endif
  }
