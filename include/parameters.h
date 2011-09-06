#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <string>
#include <complex>
#include "include/global_macro.h" //for REAL
#include "include/singleton.h"
#include "lib/Tools/InputParser/InputParser.h"

class Params{
 private:
  int no_flavors; // number of quark species
  REAL beta;      // coupling 
  REAL beta_by_three;
  REAL mass;      // mass
  REAL one_by_mass;
  REAL mass2;
  int rand_seed;  // random seed (if 0 time machine is used)
#ifdef IM_CHEM_POT
  REAL immu; // immaginary chemical potential (REAL PART)
#endif
  // fermion temporal bounday condition   =0 --> antiperiodic, else periodic
  int ferm_temp_bc;

  //=0 ordered start, =1 random start, =2 start from saved conf
  int starting_state;

  // updates parameters
  int run_update_iterations; // number of this run updates
  int therm_updates;         // number of thermalization update
  int save_interval;         // save configuration every ... updates
  int meas_interval;            // measure th observables every ... updates

  // MD parameters
  int no_md; // number of MD steps
  // =0 does not use multistep,   =1 2MN_multistep,   =2 4MN_multistep
  std::string use_multistep;  
  int gauge_scale;   // Update fermions every gauge_scale gauge updates
#ifndef __CUDACC__  // if we are not inside nvcc
  REAL epsilon;
  std::complex<REAL> ieps;
  std::complex<REAL> iepsh;
#endif

  int rand_vect; // Numbers of random vectors for measurements

  // RHMC parameters
  int max_cg;               // maximum number of iteration in CG inverter
  int no_ps;                // number of pseudofermions
  int gmp_remez_precision;  // The precision that gmp uses

  REAL inv_single_double_prec;  
  // if stopping residue <inv_single_double_prec inverter use double prec, else single

  //Rational approximations for Metropolis
  int approx_metro;
  REAL lambda_min_metro;  // rational approx valid on [lambda_min_metro, 1.0]
  REAL residue_metro;     // stopping residual for CG

  //Rational approximations for MD
  int approx_md;
  REAL lambda_min_md;  // rational approx valid on [lambda_min_metro, 1.0]
  REAL residue_md;    // stopping residual for CG

  // choice of the gpu device
  int gpu_device_to_use;
  
public:
  //maybe we need a proper constructor
  void setParams(InputParser &);

  int getNf(){return no_flavors;}

  REAL getBeta(){return beta;}
  REAL getBetaByThree(){return beta_by_three;}

  REAL getMass(){return mass;}
  REAL getMassInv(){return one_by_mass;}
  REAL getMass2(){return mass2;}

  int getRandSeed(){return rand_seed;}

#ifdef IM_CHEM_POT
  REAL getImMu(){return immu;}
#endif
  int getFermTempBC(){return ferm_temp_bc;}
  int getStartState(){return starting_state;}

  int getRunUpdateIterations(){return run_update_iterations;}
  int getThermUpdates(){return therm_updates;}
  int getSaveInterval(){return save_interval;}
  int getMeasInterval(){return meas_interval;}

  int getNumMD(){return no_md;}
  std::string getMultistep(){return use_multistep;}
  int getGaugeTimeScale(){return gauge_scale;}
#ifndef __CUDACC__  // if we are not inside nvcc
  REAL getEpsilon(){return epsilon;}
  std::complex<REAL> getIeps(){return ieps;}
  std::complex<REAL> getIepsh(){return iepsh;}
#endif

  int getRandVect(){return rand_vect;}
  
  int getMaxCG(){return max_cg;}
  int getNumPS(){return no_ps;}
  int getRemezPrecision(){return gmp_remez_precision;}

  REAL getInvSingleDoublePrec(){return inv_single_double_prec;}

  int getApproxMetro(){return approx_metro;}
  REAL getLambdaMinMetro(){return lambda_min_metro;}
  REAL getResidueMetro(){return residue_metro;}

  int getApproxMD(){return approx_md;}
  REAL getLambdaMinMD(){return lambda_min_md;}
  REAL getResidueMD(){return residue_md;}

  int getGPUDevice(){return gpu_device_to_use;}

  void listParams();
};

typedef Singleton<Params> GlobalParams;

// lattice dimensions
const int nx=8;
const int ny=8;
const int nz=8;
const int nt=4;

const int no_ps=2;                // number of pseudofermions
//used in vec3.h and fermions.h for array dimensions

const int approx_metro=15;
//used in the approximation vectors


//Rational approximations for MD
const int approx_md=8;
//used in the approximation vectors

#endif // PARAMETERS_H_
