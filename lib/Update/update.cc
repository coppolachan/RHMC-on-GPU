#ifdef USE_GPU
#include"include/gpu.h"
#endif

#include <iostream>
#include <ctime>

#include "include/global_var.h"
#include "include/tools.h"
#include "lib/Update/MD_integrators/md_integrators.h"
#include "lib/Update/momenta.h"
#include "lib/Action/fermions.h"
#include "lib/Tools/RationalApprox/rationalapprox.h"
#include "lib/Action/action.h"


// perform 1 update without metropolis test
void update_nometro()
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside update_nometro ...\n";
  #endif

  clock_t time_start, time_finish;

  // allow the use of stored min-max eigenvalues
  use_stored=0;

  time_start=clock();
  if(GlobalParams::Instance().getMultistep().compare("NO_MULTISTEP")==0)
    {
    minimum_norm2_A();
    }
  else
    {
    if(GlobalParams::Instance().getMultistep().compare("2MN_MULTISTEP")==0)
      {
      multistep_2MN();
      }
    if(GlobalParams::Instance().getMultistep().compare("4MN_MULTISTEP")==0)
      {
      multistep_4MN();
      }
    }
  time_finish=clock();
  cout << "time for update_nometro = " << ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";

  // do not allow the use of stored min-max eigenvalues
  use_stored=1;

  #ifdef DEBUG_MODE
  cout << "\tterminated update_nometro\n";
  #endif
  }



// perform 1 update with metropolis test
void update(int &acc)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside update ...\n";
  #endif

  double p1, p2, de;

  // create gaussian distributed momenta
  create_momenta();

  #ifndef PURE_GAUGE
  // create gaussian noise spinor
  create_phi(); 
  #endif

  #ifdef USE_GPU
  gpu_init1();
  #endif

  // create the chi spinor
  #ifndef PURE_GAUGE
    //chi=(M^dagM)^{alpha} phi      alpha=no_flaours/8/no_ps
  first_inv_approx_calc(GlobalParams::Instance().getResidueMetro());
  #endif

  if(update_iteration<GlobalParams::Instance().getThermUpdates())
    { 
    update_nometro();

    #ifdef USE_GPU
    gpu_end();
    #endif
    gauge_conf->save();

    acc=-1;
    }
  else
    {
    //measure initial action
    calc_action(plaq_test, 0);

    update_nometro();

    #ifndef PURE_GAUGE
      //phi=(M^dagM)^{-gamma} chi      gamma=no_flavours/4/no_ps
      last_inv_approx_calc(GlobalParams::Instance().getResidueMetro());
    #endif

    #ifdef USE_GPU
    gpu_end();
    #endif

    // measure final action
    calc_action(plaq_test, -1);
    global_sum(plaq_test, size);
    de=-plaq_test[0];   // de=action_new-action_old

    // metropolis test
    if(de<0)
      {
      cout <<"Delta S = "<<de<<"\n";
      acc=+1;

      // configuration accepted
      gauge_conf->save();
      } 
    else
      {
      p1=exp(-de);
      p2=uniform_generator();
      if(p2<p1)
        {
        cout <<"Delta S = "<<de<<"\n";
        acc=+1;

        // configuration accepted
        gauge_conf->save();
        }
      else
        {
        cout <<"Delta S = "<<de<<"\n";
        acc=0;

        // configuration rejected
        gauge_conf->copy_saved();
        }
      }

    }

  #ifdef DEBUG_MODE
  cout << "\tterminated update\n";
  #endif
  }

