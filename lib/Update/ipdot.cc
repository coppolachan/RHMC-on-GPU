#include <iostream>
#include "include/global_var.h"
#include "lib/Update/ipdot.h"
#include "lib/Action/fermionforce.h"

#ifdef USE_GPU
#include"lib/Cuda/cuda_init.h"
#include"lib/Cuda/cuda_deriv.h"
#include"lib/Tools/Packer/packer.h"
#endif



// base constructor
Ipdot::Ipdot(void)
 {
 for(long int i=0; i<no_links; i++)
    {
    ipdot[i].zero();
    }
 }


// calculate the gauge part of ipdot
void calc_ipdot_gauge(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside calc_ipdot_gauge ..."<<endl;
 #endif

 #ifdef TIMING_CUDA_CPP
 clock_t time_start, time_finish;
 time_start=clock();
 #endif

 #ifdef USE_GPU 
   cuda_gauge_deriv(0);
 #else
   long int r;
   Su3 aux;

   calc_staples(); 
   for(r=0; r<no_links; r++)
      {
      // ipdot = traceless anti-hermitian part of beta_by_three*(u_work*staple) 
      aux=(gauge_conf->u_work[r]);
      aux*=(beta_by_three);         // PLUS SIGN DUE TO STAGGERED PHASES
      aux*=(gauge_staples->staples[r]);
      aux.ta();
      (gauge_ipdot->ipdot[r])=aux;
      }
 #endif

 #ifdef TIMING_CUDA_CPP
 time_finish=clock();
 cout << "time for calc_ipdot_gauge = " << ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
 #endif

 #ifndef USE_GPU
   #ifdef PARAMETER_TEST
   int mu;
   for(r=0; r<size; r++)
      {
      d_vector1[r]=0.0;
      for(mu=0; mu<4; mu++)
         {
         d_vector1[r]+=(gauge_ipdot->ipdot[r+mu*size]).l2norm2();
         }
      }
   global_sum(d_vector1, size);

   cout << "L2 norm of GAUGE force = " << sqrt(d_vector1[0])<<endl;
   #endif
 #endif

 #ifdef DEBUG_MODE
 cout << "\tterminated calc_ipdot_gauge"<<endl;
 #endif
 }


// calculate the fermionic part of ipdot
void calc_ipdot_fermion(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside calc_ipdot_fermion ..."<<endl;
 #endif

 #ifdef TIMING_CUDA_CPP
 clock_t time_start, time_finish;
 time_start=clock();
 #endif

 #ifdef USE_GPU 
   fermionforce();
 #else
   long int r;
   Su3 aux;

   // initialize to zero
   for(r=0; r<no_links; r++)
      {
      (gauge_ipdot->ipdot[r]).zero();
      }

   // add fermionic term
   fermionforce();

   // multiply by u_work and take TA
   for(r=0; r<no_links; r++)
      {
      aux=(gauge_conf->u_work[r]);
      aux*=(gauge_ipdot->ipdot[r]);
      aux.ta();
      (gauge_ipdot->ipdot[r])=aux;
      }
 #endif

 #ifdef TIMING_CUDA_CPP
 time_finish=clock();
 cout << "time for calc_ipdot_fermion = " << ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
 #endif

 #ifdef PARAMETER_TEST
  #ifndef USE_GPU
    int mu;
    for(r=0; r<size; r++)
       {
       d_vector1[r]=0.0;
       for(mu=0; mu<4; mu++)
          {
          d_vector1[r]+=(gauge_ipdot->ipdot[r+mu*size]).l2norm2();
          }
       }
    global_sum(d_vector1, size);

    cout << "L2 norm of FERMION force = " << sqrt(d_vector1[0])<<endl;
    #endif
 #endif

 #ifdef DEBUG_MODE
 cout << "\tterminated calc_ipdot_fermion"<<endl;
 #endif
 }



// calculate the complete ipdot
void calc_ipdot(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside calc_ipdot ..."<<endl;
 #endif

 #ifdef TIMING_CUDA_CPP
 clock_t time_start, time_finish;
 time_start=clock();
 #endif

 #ifdef USE_GPU 
   fermionforce();
   cuda_gauge_deriv(1);  // add gauge part
 #else
   long int r;
   Su3 aux;

   calc_staples(); 

   // add gauge_part
   for(r=0; r<no_links; r++)
      {
      (gauge_ipdot->ipdot[r])=beta_by_three*(gauge_staples->staples[r]);
      }

   // add fermionic term
   fermionforce();

   // multiply by u_work and take TA
   for(r=0; r<no_links; r++)
      {
      aux=(gauge_conf->u_work[r]);
      aux*=(gauge_ipdot->ipdot[r]);
      aux.ta();
      (gauge_ipdot->ipdot[r])=aux;
      }
 #endif

 #ifdef TIMING_CUDA_CPP
 time_finish=clock();
 cout << "time for calc_ipdot = " << ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
 #endif

 #ifdef DEBUG_MODE
 cout << "\tterminated calc_ipdot"<<endl;
 #endif
 }
