// MULTISTEP VERSION of minimum_norm2_B

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
//
// See reference hep-lat/0506011 Urbach et al. for the multiple time scale 
//
// p->p+a dS/dq=p+ia(-i dS/dq)=p+ia*ipdot

#include <iostream>
#include <complex>
#include "lib/Update/MD_integrators/md_integrators.h"
#include "include/global_macro.h"
#include "lib/Update/momenta.h"
#include "lib/Update/ipdot.h"

using namespace std;

void multistep_2MN_gauge(REAL scale)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside multistep_2MN_gauge ..."<<endl;
 #endif

 int md;
 const REAL lambda=0.1931833275037836; // Omelyan Et Al.
 complex<REAL> temp;

 // Step for the P
 // P' = P - l*dt*dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(scale*lambda,0.0);
 momenta_sum_multiply(temp);

 for(md=1; md<GlobalParams::Instance().getGaugeTimeScale(); md++)
    {
    // Step for the Q
    // Q' = exp[dt/2 *i P] Q
    temp=GlobalParams::Instance().getIepsh()*complex<REAL>(scale,0.0);
    conf_left_exp_multiply(temp);

    // Step for the P
    // P' = P - (1-2l)*dt*dS/dq
    calc_ipdot_gauge();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>((1.0-2.0*lambda)*scale,0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt/2 *i P] Q
    temp=GlobalParams::Instance().getIepsh()*complex<REAL>(scale,0.0);
    conf_left_exp_multiply(temp);

    // Step for the P
    // P' = P - 2l*dt*dS/dq
    calc_ipdot_gauge();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>(2.0*lambda*scale,0.0);
    momenta_sum_multiply(temp);
    }

 // Step for the Q
 // Q' = exp[dt/2 *i P] Q
 temp=GlobalParams::Instance().getIepsh()*complex<REAL>(scale,0.0);
 conf_left_exp_multiply(temp);

 // Step for the P
 // P' = P - (1-2l)*dt*dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>((1.0-2.0*lambda)*scale,0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt/2 *i P] Q
 temp=GlobalParams::Instance().getIepsh()*complex<REAL>(scale,0.0);
 conf_left_exp_multiply(temp);

 // Step for the P
 // P' = P - l*dt*dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda*scale,0.0);
 momenta_sum_multiply(temp);

 #ifdef DEBUG_MODE
 cout << "\tterminated multistep_2MN_gauge"<<endl;
 #endif
 }






void multistep_2MN(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside multistep_2MN ..."<<endl;
 #endif

 int md;
 const REAL lambda=0.1931833275037836; // Omelyan Et Al.
 const REAL gs=0.5/(REAL) GlobalParams::Instance().getGaugeTimeScale();
 complex<REAL> temp;
 

 // Step for the P
 // P' = P - l*dt*dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda,0.0);
 momenta_sum_multiply(temp);

 for(md=1; md<GlobalParams::Instance().getNumMD(); md++)
    {
    // Step for the Q
    // Q' = exp[dt/2 *i P] Q
    multistep_2MN_gauge(gs);

    // Step for the P
    // P' = P - (1-2l)*dt*dS/dq
    calc_ipdot_fermion();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>((1.0-2.0*lambda),0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt/2 *i P] Q
    multistep_2MN_gauge(gs);

    // Step for the P
    // P' = P - 2l*dt*dS/dq
    calc_ipdot_fermion();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>(2.0*lambda,0.0);
    momenta_sum_multiply(temp);
    }

 // Step for the Q
 // Q' = exp[dt/2 *i P] Q
 multistep_2MN_gauge(gs);

 // Step for the P
 // P' = P - (1-2l)*dt*dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>((1.0-2.0*lambda),0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt/2 *i P] Q
 multistep_2MN_gauge(gs);

 // Step for the P
 // P' = P - l*dt*dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda,0.0);
 momenta_sum_multiply(temp);

 #ifdef DEBUG_MODE
 cout << "\tterminated multistep_2MN"<<endl;
 #endif
 }
