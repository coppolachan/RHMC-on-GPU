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

#include <iostream>
#include "include/global_const.h"
#include "lib/Update/MD_integrators/md_integrators.h"
#include "lib/Update/momenta.h"
#include "lib/Update/ipdot.h"

void leapfrog()
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside leapfrog ..."<<endl;
  #endif

  int md;

   // Step for the Q
   // Q -> exp(i dt/2 P) Q
  conf_left_exp_multiply(GlobalParams::Instance().getIepsh());

  for(md=1; md<GlobalParams::Instance().getNumMD(); md++)
      {
      // Step for the P
      // P -> P - i dt dS/dq
      #ifndef PURE_GAUGE
        calc_ipdot();
      #else
        calc_ipdot_gauge();
      #endif
      momenta_sum_multiply(GlobalParams::Instance().getIeps());

      // Step for the Q
      // Q -> exp(i dt P) Q
      conf_left_exp_multiply(GlobalParams::Instance().getIeps());
      }

  // Step for the P
  // P -> P - i dt dS/dq
  #ifndef PURE_GAUGE
    calc_ipdot();
  #else
    calc_ipdot_gauge();
  #endif
  momenta_sum_multiply(GlobalParams::Instance().getIeps());

  // Step for the Q
  // Q -> exp(i dt/2 P) Q
  conf_left_exp_multiply(GlobalParams::Instance().getIepsh());

  #ifdef DEBUG_MODE
  cout << "\tterminated leapfrog"<<endl;
  #endif
  }
