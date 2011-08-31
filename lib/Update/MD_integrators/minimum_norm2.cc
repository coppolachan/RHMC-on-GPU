// 2nd order Minimum Norm integrator (2MN)
// See reference hep-lat/0505020 Takaishi, De Forcrand
//
// Scheme (needs two force calculations per step):
// 2MN(dt) = exp (l * dt * p) exp (- dt/2 * dS/dq) * ...
//           exp ((1- 2l) * dt * p) exp (- dt/2 * dS/dq)  exp (l * dt * p)
// p       : momenta
// - dS/sq : force contibution
// dt      : integration step
// l       : lambda parameter (1/6 Sexton Weingarten, 0.193... Omelyan et al)
//    
// Total scheme:
// [2MN]^N
// N       : number of Molecular Dynamics steps
//
// p->p+a dS/dq=p+ia(-i dS/dq)=p+ia*ipdot

#include <iostream>
#include <complex>
#include "include/global_macro.h"
#include "include/global_var.h"
#include "lib/Update/MD_integrators/md_integrators.h"
#include "lib/Update/momenta.h"
#include "lib/Update/ipdot.h"

using namespace std;

void minimum_norm2_A(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside minimum_norm2_A ..."<<endl;
 #endif

 int md;
 const REAL lambda=0.1931833275037836; // Omelyan Et Al.
 complex<REAL> temp;

 // First step for the Q 
 // Q -> exp(i dt lambda P) Q
 temp=ieps*complex<REAL>(lambda,0.0);
 conf_left_exp_multiply(temp);

 for(md=1; md<no_md; md++)
    {
    // Step for the P
    // P' = P - dt/2 dS/dq
    #ifndef PURE_GAUGE
      calc_ipdot();
    #else
      calc_ipdot_gauge();
    #endif
    momenta_sum_multiply(-iepsh);

    // Step for the Q
    // Q' = exp[(1-2l) * dt *i P] Q
    temp=ieps*complex<REAL>((1.0-2.0*lambda),0.0);
    conf_left_exp_multiply(temp);

    // Step for the P
    // P' = P - dt/2 dS/dq
    #ifndef PURE_GAUGE
      calc_ipdot();
    #else
      calc_ipdot_gauge();
    #endif
    momenta_sum_multiply(-iepsh);

    // Step for the Q
    // Q' = exp[2l * dt *i P] Q
    temp=ieps*complex<REAL>(2.0*lambda,0.0);
    conf_left_exp_multiply(temp);
    }

 // Step for the P
 // P' = P - dt/2 dS/dq
 #ifndef PURE_GAUGE
   calc_ipdot();
 #else
   calc_ipdot_gauge();
 #endif
 momenta_sum_multiply(-iepsh);

 // Step for the Q
 // Q' = exp[(1-2l) * dt *i P] Q
 temp=ieps*complex<REAL>((1.0-2.0*lambda),0.0);
 conf_left_exp_multiply(temp);

 // Step for the P
 // P' = P - dt/2 dS/dq
 #ifndef PURE_GAUGE
   calc_ipdot();
 #else
   calc_ipdot_gauge();
 #endif
 momenta_sum_multiply(-iepsh);

 // Step for the Q
 // Q' = exp[l * dt *i P] Q
 temp=ieps*complex<REAL>(lambda,0.0);
 conf_left_exp_multiply(temp);

 #ifdef DEBUG_MODE
 cout << "\tterminated minimum_norm2_A"<<endl;
 #endif
 }





// 2nd order Minimum Norm integrator (2MN)
// See reference hep-lat/0505020 Takaishi, De Forcrand
//
// Scheme (needs three force calculations per step):
// 2MN(dt) = exp (-l * dt * dS/dq) exp (dt/2 * p) * ...
//           exp (-(1- 2l) * dt * dS/dq) exp (dt/2 * p)  exp (-l * dt * dS/dq)
// p       : momenta
// - dS/sq : force contibution
// dt      : integration step
// l       : lambda parameter (1/6 Sexton Weingarten, 0.193... Omelyan et al)
//    
// Total scheme:
// [2MN]^N
// N       : number of Molecular Dynamics steps

void minimum_norm2_B(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside minimum_norm2_B ..."<<endl;
 #endif

 int md;
 const REAL lambda=0.1931833275037836; // Omelyan Et Al.
 complex<REAL> temp;

 // Step for the P
 // P' = P - l*dt*dS/dq
 #ifndef PURE_GAUGE
   calc_ipdot();
 #else
   calc_ipdot_gauge();
 #endif
 temp=-ieps*complex<REAL>(lambda,0.0);
 momenta_sum_multiply(temp);

 for(md=1; md<no_md; md++)
    {
    // Step for the Q
    // Q' = exp[dt/2 *i P] Q
    conf_left_exp_multiply(iepsh);
    gauge_conf->unitarize_with_eta(); 

    // Step for the P
    // P' = P - (1-2l)*dt*dS/dq
    #ifndef PURE_GAUGE
      calc_ipdot();
    #else
      calc_ipdot_gauge();
    #endif
    temp=-ieps*complex<REAL>((1.0-2.0*lambda),0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt/2 *i P] Q
    conf_left_exp_multiply(iepsh);

    // Step for the P
    // P' = P - 2l*dt*dS/dq
    #ifndef PURE_GAUGE
      calc_ipdot();
    #else
      calc_ipdot_gauge();
    #endif
    temp=-ieps*complex<REAL>(2.0*lambda,0.0);
    momenta_sum_multiply(temp);
    }

 // Step for the Q
 // Q' = exp[dt/2 *i P] Q
 conf_left_exp_multiply(iepsh);

 // Step for the P
 // P' = P - (1-2l)*dt*dS/dq
 #ifndef PURE_GAUGE
   calc_ipdot();
 #else
   calc_ipdot_gauge();
 #endif
 temp=-ieps*complex<REAL>((1.0-2.0*lambda),0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt/2 *i P] Q
 conf_left_exp_multiply(iepsh);

 // Step for the P
 // P' = P - l*dt*dS/dq
 #ifndef PURE_GAUGE
   calc_ipdot();
 #else
   calc_ipdot_gauge();
 #endif
 temp=-ieps*complex<REAL>(lambda,0.0);
 momenta_sum_multiply(temp);

 #ifdef DEBUG_MODE
 cout << "\tterminated minimum_norm2_B"<<endl;
 #endif
 }
