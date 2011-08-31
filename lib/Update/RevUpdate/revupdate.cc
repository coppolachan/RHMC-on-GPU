#include "include/global_macro.h" // defines REVERSIBILITY_TEST


#ifdef REVERSIBILITY_TEST

//#include "revupdate.h"
#include <iostream>
#include "lib/Action/su3.h"
#include "lib/Update/momenta.h"
#include "lib/Action/fermions.h"
#include "include/tools.h"
#include "include/update.h"

#ifdef USE_GPU
#include"include/gpu.h"
#endif

// perform update, reverse momenta and come back
void rev_update()
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside rev_update ..."<<endl;
  #endif
  Su3 aux;
  int mu;
  double diff1, diff2;
  long int i, pos;  

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
    first_inv_approx_calc(residue_metro);
  #endif
 
  // save initial momenta
  save_momenta();

  update_nometro();

  // reverse momenta
  reverse_momenta();

  update_nometro();

  #ifndef PURE_GAUGE
    //phi=(M^dagM)^{-gamma} chi      gamma=no_flavours/4/no_ps
    first_inv_approx_calc(residue_metro);
  #endif

  #ifdef USE_GPU
  gpu_end();
  #endif

  for(i=0; i<size; i++)
     {
     d_vector1[i]=0.0;
     d_vector2[i]=0.0;
     for(mu=0; mu<4; mu++)
        {
        pos=i+mu*size;       

        // gauge part
        aux=(gauge_conf->u_work[pos]);
        aux-=(gauge_conf->u_save[pos]);
        d_vector1[i]+=(double)aux.l2norm2();

        // momenta part
        aux=(gauge_momenta->momenta[pos]);
        aux+=(gauge_momenta_save->momenta[pos]);    // + because momenta have to be (-1)*initial momenta
        d_vector2[i]+=(double)aux.l2norm2();
        }
     }

  global_sum(d_vector1,size);
  diff1=sqrt(d_vector1[0])*sqrt(inv_size*0.25*0.125);  // 4 directions, 8 components
  cout << "Delta Conf / d.o.f. = " << diff1<<"\n";

  global_sum(d_vector2,size);
  diff2=sqrt(d_vector2[0])*sqrt(inv_size*0.25*0.125);  // 4 directions, 8 components
  cout << "Delta Momenta / d.o.f. = " << diff2<< endl;

  #ifdef DEBUG_MODE
  cout << "\tterminated rev_update"<<endl;
  #endif
  }
#endif
