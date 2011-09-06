// 4th order Minimum Norm integrator (4MN)
// See reference hep-lat/0505020 Takaishi, De Forcrand
//
// Scheme (needs five force calculations per step):
// 4MN(dt) = exp(-theta*dt*dS/dq) exp(dt*rho*p) exp(-lambda*dt*dS/dq) *
//           exp(mu*dt*p) exp(-dt/2 *(1-2(lambda+theta))*dS/dq)*
//           exp(dt*(1-2(mu+rho))*p) exp(-dt/2 *(1-2(lambda+theta))*dS/dq)*
//           exp(mu*dt*p) exp(-lambda*dt*dS/dq) exp(dt*rho*p) exp(-theta*dt*dS/dq) 
// p       : momenta
// - dS/sq : force contibution
// dt      : integration step
// theta, rho, lambda, mu : parameters
//    
// Total scheme:
// [4MN]^N
// N       : number of Molecular Dynamics steps
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

void multistep_4MN_gauge(REAL scale)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside multistep_4MN_gauge ..."<<endl;
 #endif

 int md;
 const REAL theta = 0.08398315262876693;
 const REAL rho = 0.2539785108410595;
 const REAL lambda = 0.6822365335719091;
 const REAL mu = -0.03230286765269967;
 complex<REAL> temp;

 // Step for the P
 // P' = P - theta*dt*dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(scale*theta,0.0);
 momenta_sum_multiply(temp);

 for(md=1; md<GlobalParams::Instance().getGaugeTimeScale(); md++)
    {
    // Step for the Q
    // Q' = exp[dt * rho *i P] Q
    temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*rho,0.0);
    conf_left_exp_multiply(temp);

    // Step for the P
    // P' = P - lambda*dt*dS/dq
    calc_ipdot_gauge();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda*scale,0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt * mu *i P] Q
    temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*mu,0.0);
    conf_left_exp_multiply(temp);

    // Step for the P
    // P' = P - dt/2 * [1-2(lambda+theta)] * dS/dq
    calc_ipdot_gauge();
    temp=-GlobalParams::Instance().getIepsh()*complex<REAL>((1.0-2.0*(lambda+theta))*scale,0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt * mu *i P] Q
    temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*(1.0-2.0*(mu+rho)),0.0);
    conf_left_exp_multiply(temp);

    // Step for the P
    // P' = P - dt/2 * [1-2(lambda+theta)] * dS/dq
    calc_ipdot_gauge();
    temp=-GlobalParams::Instance().getIepsh()*complex<REAL>((1.0-2.0*(lambda+theta))*scale,0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt * mu *i P] Q
    temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*mu,0.0);
    conf_left_exp_multiply(temp);

    // Step for the P
    // P' = P - lambda*dt*dS/dq
    calc_ipdot_gauge();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda*scale,0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt * rho *i P] Q
    temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*rho,0.0);
    conf_left_exp_multiply(temp);

    // Step for the P
    // P' = P - theta*dt*dS/dq
    calc_ipdot_gauge();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>(2*scale*theta,0.0);
    momenta_sum_multiply(temp);
    }

 // Step for the Q
 // Q' = exp[dt * rho *i P] Q
 temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*rho,0.0);
 conf_left_exp_multiply(temp);

 // Step for the P
 // P' = P - lambda*dt*dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda*scale,0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt * mu *i P] Q
 temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*mu,0.0);
 conf_left_exp_multiply(temp);

 // Step for the P
 // P' = P - dt/2 * [1-2(lambda+theta)] * dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIepsh()*complex<REAL>((1.0-2.0*(lambda+theta))*scale,0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt * mu *i P] Q
 temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*(1.0-2.0*(mu+rho)),0.0);
 conf_left_exp_multiply(temp);

 // Step for the P
 // P' = P - dt/2 * [1-2(lambda+theta)] * dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIepsh()*complex<REAL>((1.0-2.0*(lambda+theta))*scale,0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt * mu *i P] Q
 temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*mu,0.0);
 conf_left_exp_multiply(temp);

 // Step for the P
 // P' = P - lambda*dt*dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda*scale,0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt * rho *i P] Q
 temp=GlobalParams::Instance().getIeps()*complex<REAL>(scale*rho,0.0);
 conf_left_exp_multiply(temp);

 // Step for the P
 // P' = P - theta*dt*dS/dq
 calc_ipdot_gauge();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(scale*theta,0.0);
 momenta_sum_multiply(temp);

 #ifdef DEBUG_MODE
 cout << "\tterminated multistep_4MN_gauge"<<endl;
 #endif
 }






void multistep_4MN(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside multistep_4MN ..."<<endl;
 #endif

 int md;
 const REAL theta = 0.08398315262876693;
 const REAL rho = 0.2539785108410595;
 const REAL lambda = 0.6822365335719091;
 const REAL mu = -0.03230286765269967;
 const REAL gs=1.0/(REAL) GlobalParams::Instance().getGaugeTimeScale();
 complex<REAL> temp;
 

 // Step for the P
 // P' = P - theta*dt*dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(theta,0.0);
 momenta_sum_multiply(temp);

 for(md=1; md<GlobalParams::Instance().getNumMD(); md++)
    {
    // Step for the Q
    // Q' = exp[dt * rho *i P] Q
    multistep_4MN_gauge(rho*gs);

    // Step for the P
    // P' = P - lambda*dt*dS/dq
    calc_ipdot_fermion();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda,0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt * mu *i P] Q
    multistep_4MN_gauge(mu*gs);

    // Step for the P
    // P' = P - dt/2 * [1-2(lambda+theta)] * dS/dq
    calc_ipdot_fermion();
    temp=-GlobalParams::Instance().getIepsh()*complex<REAL>((1.0-2.0*(lambda+theta)),0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt * mu *i P] Q
    multistep_4MN_gauge(gs*(1.0-2.0*(mu+rho)));

    // Step for the P
    // P' = P - dt/2 * [1-2(lambda+theta)] * dS/dq
    calc_ipdot_fermion();
    temp=-GlobalParams::Instance().getIepsh()*complex<REAL>((1.0-2.0*(lambda+theta)),0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt * mu *i P] Q
    multistep_4MN_gauge(mu*gs);

    // Step for the P
    // P' = P - lambda*dt*dS/dq
    calc_ipdot_fermion();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda,0.0);
    momenta_sum_multiply(temp);

    // Step for the Q
    // Q' = exp[dt * rho *i P] Q
    multistep_4MN_gauge(rho*gs);

    // Step for the P
    // P' = P - theta*dt*dS/dq
    calc_ipdot_fermion();
    temp=-GlobalParams::Instance().getIeps()*complex<REAL>(2*theta,0.0);
    momenta_sum_multiply(temp);
    }

 // Step for the Q
 // Q' = exp[dt * rho *i P] Q
 multistep_4MN_gauge(rho*gs);

 // Step for the P
 // P' = P - lambda*dt*dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda,0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt * mu *i P] Q
 multistep_4MN_gauge(mu*gs);

 // Step for the P
 // P' = P - dt/2 * [1-2(lambda+theta)] * dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIepsh()*complex<REAL>((1.0-2.0*(lambda+theta)),0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt * mu *i P] Q
 temp=GlobalParams::Instance().getIeps()*complex<REAL>((1.0-2.0*(mu+rho))*gs,0.0);
 multistep_4MN_gauge((1.0-2.0*(mu+rho))*gs);

 // Step for the P
 // P' = P - dt/2 * [1-2(lambda+theta)] * dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIepsh()*complex<REAL>((1.0-2.0*(lambda+theta)),0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt * mu *i P] Q
 multistep_4MN_gauge(mu*gs);

 // Step for the P
 // P' = P - lambda*dt*dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(lambda,0.0);
 momenta_sum_multiply(temp);

 // Step for the Q
 // Q' = exp[dt * rho *i P] Q
 multistep_4MN_gauge(rho*gs);

 // Step for the P
 // P' = P - theta*dt*dS/dq
 calc_ipdot_fermion();
 temp=-GlobalParams::Instance().getIeps()*complex<REAL>(theta,0.0);
 momenta_sum_multiply(temp);

 #ifdef DEBUG_MODE
 cout << "\tterminated multistep_4MN"<<endl;
 #endif
 }
