#ifndef MD_INTEGRATORS_H_
#define MD_INTEGRATORS_H_

// Basic 2nd order LeapFrog update (2LF)
// Scheme:
// 2LF(dt) = exp (dt/2 * p) exp (- dt * dS/dq) exp (dt/2 * p)
// p       : momenta
// - dS/dq : force contibution
// dt      : integration step
//
// Total: (LFN)^no_md
//
// p->p+a dS/dq=p+ia(-i dS/dq)=p+ia*ipdot

void leapfrog();



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

void minimum_norm2_A(void);


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

void multistep_2MN(void);


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

void multistep_4MN(void);

#endif //MD_INTEGRATORS_H_
