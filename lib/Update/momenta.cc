#include <iostream>
#include "include/global_var.h"
#include "lib/Update/momenta.h"

#ifdef USE_GPU
#include"lib/Cuda/cuda_exponentiate.h"
#include"lib/Cuda/cuda_mom_sum_mul.h"
#include"lib/Cuda/cuda_init.h"
#include"lib/Tools/Packer/packer.h"
#endif



// base constructor
Momenta::Momenta(void)
 {
 for(long int i=0; i<no_links; i++)
    {
    momenta[i].zero();
    }
 }


// create gaussian distributed momenta
void create_momenta(void)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside create_momenta ..."<<endl;
  #endif

  for(long int i=0; i<no_links; i++)
     {
     (gauge_momenta->momenta[i]).gauss();
     }
  #ifdef DEBUG_MODE
  cout << "\tterminated create_momenta"<<endl;
  #endif
  } 


// u_work[i] -> exp(p*momenta[i])*u_work[i]
void conf_left_exp_multiply(const complex<REAL> p)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside conf_left_exp_multiply ..."<<endl;
  #endif

  #ifdef TIMING_CUDA_CPP
  clock_t time_start, time_finish;
  time_start=clock();
  #endif

  #ifdef USE_GPU
    cuda_exp_su3(imag(p));
  #else
    Su3 aux;
    long int i;

    for(i=0; i<no_links; i++)
       {
       aux=(gauge_momenta->momenta[i]);
       aux*=p;
       aux.exp();
       aux*=(gauge_conf->u_work[i]);
       (gauge_conf->u_work[i])=aux;
       }
    gauge_conf->unitarize_with_eta(); 
  #endif

  #ifdef TIMING_CUDA_CPP
  time_finish=clock();
  cout << "time for conf_left_exp_multiply = " << ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
  #endif

  #ifdef DEBUG_MODE
  cout << "\tterminated conf_left_exp_multiply"<<endl;
  #endif
  }


// momenta[i]+=p*ipdot_loc[i]
void momenta_sum_multiply(const complex<REAL> p)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside momenta_sum_multiply ..."<<endl;
  #endif

  #ifdef TIMING_CUDA_CPP
  clock_t time_start, time_finish;
  time_start=clock();
  #endif

  #ifdef USE_GPU
    cuda_momenta_sum_multiply(imag(p));
  #else
    long int i;
    Su3 aux;

    for(i=0; i<no_links; i++)
       {
       aux=(gauge_ipdot->ipdot[i]);
       aux*=p;
       (gauge_momenta->momenta[i])+=aux;
       }
  #endif

  #ifdef TIMING_CUDA_CPP
  time_finish=clock();
  cout << "time for momenta_sum_multiply = " << ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
  #endif

  #ifdef DEBUG_MODE
  cout << "\tterminated momenta_sum_multiply"<<endl;
  #endif
  }


#ifdef REVERSIBILITY_TEST
// reverse momenta
// to be used in reversibility tests
void reverse_momenta(void)
  {
  #ifndef USE_GPU
  for(long int i=0; i<no_links; i++)
     {
     (gauge_momenta->momenta[i])*=(-1.0);
     }
  #else
  cuda_get_momenta();
  smartunpack_thmatrix(gauge_momenta, momenta_packed);
  for(long int i=0; i<no_links; i++)
     {
     (gauge_momenta->momenta[i])*=(-1.0);
     }
  smartpack_thmatrix(momenta_packed, gauge_momenta);
  cuda_put_momenta();
  #endif
  } 


// to be used in reversibility tests
void save_momenta(void)
  {
  for(long int i=0; i<no_links; i++)
     {
     (gauge_momenta_save->momenta[i])=(gauge_momenta->momenta[i]);
     }
  } 
#endif

