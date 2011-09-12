//Definitions of some global variables

#ifndef GLOBAL_VAR_H_
#define GLOBAL_VAR_H_

#include <cmath>
#include <complex>
#include "include/global_macro.h"
#include "include/configuration.h"
#include "lib/Tools/RationalApprox/rationalapprox.h"
#include "lib/Action/staples.h"
#include "lib/Update/momenta.h"
#include "lib/Update/ipdot.h"
#include "lib/Action/fermions.h"

using namespace std;

#ifdef DEFINE_GLOBALS
#define GLOBAL
#else 
#define GLOBAL extern
#endif


GLOBAL int update_iteration;

// eigenvalues
static REAL min_stored=0.0;
static REAL max_stored=0.0;
static int use_stored=1;  // when =0 stored eigenvalues are used

// normalized coefficients for rational approximations
GLOBAL RationalApprox *first_inv_approx_norm_coeff;
GLOBAL RationalApprox *md_inv_approx_norm_coeff;
GLOBAL RationalApprox *last_inv_approx_norm_coeff;
GLOBAL RationalApprox *meas_inv_coeff;

// next neighbours
GLOBAL long int **nnp;    
GLOBAL long int **nnm;   

// configuration
GLOBAL Conf *gauge_conf;

// staples
GLOBAL Staples *gauge_staples;

// momenta
GLOBAL Momenta *gauge_momenta;
#ifdef REVERSIBILITY_TEST
GLOBAL Momenta *gauge_momenta_save;
#endif

// momenta derivative times complex_I
GLOBAL Ipdot *gauge_ipdot;

// vectors for global sum
GLOBAL double *d_vector1;
GLOBAL double *d_vector2;
GLOBAL complex<double> *c_vector1;
GLOBAL complex<double> *c_vector2;
GLOBAL double *plaq_test;   // this has to be used only in action calculations

// staggered phases
GLOBAL int *eta;

// fermions
GLOBAL MultiFermion *fermion_phi;
GLOBAL MultiFermion *fermion_chi;

// auxiliary global fermions to be used in inverters
GLOBAL Fermion *loc_r;
GLOBAL Fermion *loc_h;
GLOBAL Fermion *loc_s;
GLOBAL Fermion *loc_p;

// to be used in shifted inverter and fermion force calculation
class ShiftFermion;
GLOBAL ShiftFermion *p_shiftferm;
GLOBAL ShiftMultiFermion *fermion_shiftmulti;

// to be used in meas_chiral
GLOBAL Fermion *chi_e;
GLOBAL Fermion *chi_o;
GLOBAL Fermion *phi_e;
GLOBAL Fermion *phi_o;
GLOBAL Fermion *rnd_e;
GLOBAL Fermion *rnd_o;

#ifdef USE_GPU
GLOBAL float *simple_fermion_packed;
GLOBAL float *chi_packed;
GLOBAL float *psi_packed;
GLOBAL float *gauge_field_packed;
GLOBAL float *ipdot_packed;
GLOBAL float *momenta_packed;
GLOBAL int *shift_table;
#endif


#endif //GLOBAL_VAR_H_
