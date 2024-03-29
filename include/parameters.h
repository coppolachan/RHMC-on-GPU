#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "include/global_macro.h" //for REAL

// lattice dimensions
const int nx=8;
const int ny=8;
const int nz=8;
const int nt=4;


// run parameters
const int no_flavours=2; // number of quark species
const REAL beta=5.264;
const REAL mass=0.01;

// random seed (if 0 time machine is used)
const int rand_seed=0;


// immaginary chemical potential (REAL PART)
#ifdef IM_CHEM_POT
const REAL immu=0.2481858;
#endif


// fermion temporal bounday condition   =0 --> antiperiodic, else periodic
const int ferm_temp_bc=0;               


// start parameter
const int start=0;  //=0 ordered start, =1 random start, =2 start from saved conf


// updates parameters
const int run_update_iterations=100; // number of this run updates
const int therm_updates=0; // number of thermalization update
const int save_conf_every=20; // save configuration every ... updates


// measure th observables every ... updates
const int meas_every=2;


// MD parameters
const int no_md=12; // number of MD steps
const int use_multistep=1;  // =0 does not use multistep,   =1 2MN_multistep,   =2 4MN_multistep
const int gauge_scale=10;   // Update fermions every gauge_scale gauge updates


// Numbers of random vectors for measurements
const int rand_vect=1;


// RHMC parameters
const int max_cg=5000;             // maximum number of iteration in CG inverter
const int no_ps=2;                 // number of pseudofermions
const int gmp_remez_precision=100; // The precision that gmp uses

const REAL inv_single_double_prec=1.0e-4;  
// if stopping residue <inv_single_double_prec inverter use double prec, else single


//Rational approximations for Metropolis
const int approx_metro=15;
const REAL lambda_min_metro=1.0e-5;  // rational approx valid on [lambda_min_metro, 1.0]
const REAL residue_metro=1.0e-9;    // stopping residual for CG

//Rational approximations for MD
const int approx_md=8;
const REAL lambda_min_md=1.0e-5;  // rational approx valid on [lambda_min_metro, 1.0]
const REAL residue_md=1.0e-3;    // stopping residual for CG

// choice of the gpu device
const int gpu_device_to_use=0;


#endif // PARAMETERS_H_
