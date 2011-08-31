#include <iostream>
#include "include/global_var.h"
#include "include/tools.h"

#ifdef USE_GPU
#include"lib/Cuda/cuda_find_min_max.h"
#include"lib/Tools/Packer/packer.h"
#endif

// find the maximum eigenvalue of the fermion matrix
// use loc_h, loc_p, loc_r
void findmaxeig(REAL &max)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside findmaxeig..."<<endl;
  #endif

  int loop_count;
  long int i;
  REAL norm, inorm, old_norm;
  Vec3 vr_1, vr_2, vr_3;

  // starting gauss vector p
  loc_p->gauss(); 
  for(i=0; i<sizeh; i++)
     {
     d_vector1[i]=(loc_p->fermion[i]).l2norm2();
     }
  global_sum(d_vector1,sizeh);
  norm=sqrt(d_vector1[0]);

  loop_count=0;
  do                   // loop start
    {
    // normalize  p 
    // and r=p
    inorm=1.0/norm;
    for(i=0; i<sizeh; i++)
       {
       vr_1=(loc_p->fermion[i]);
       vr_1*=inorm;
       (loc_p->fermion[i])=vr_1;
       (loc_r->fermion[i])=vr_1;  // r=p
       }
    old_norm=norm;

    Doe(loc_h,loc_p); 
    Deo(loc_p,loc_h);   // p=DeoDoe r
    for(i=0; i<sizeh; i++)
       {
       vr_1=(loc_r->fermion[i]);
       vr_2=(loc_p->fermion[i]);
       vr_3=mass2*vr_1-vr_2;
       (loc_p->fermion[i])=vr_3;   // p=(M^dag M)r
       d_vector1[i]=vr_3.l2norm2();
       }
    global_sum(d_vector1,sizeh);
    norm=sqrt(d_vector1[0]); 

    old_norm=fabs(old_norm-norm);
    old_norm/=norm;
    loop_count++;
    } while(old_norm>1.0e-5);    // loop end
  
  max=norm;
  
  #ifdef DEBUG_MODE
  cout << "\tterminated findmaxeig in "<<loop_count<<" iterations"<<endl;
  #endif
  }



// find the minimum eigenvalue of the fermion matrix
// use loc_h, loc_p, loc_r
void findmineig(REAL &min, const REAL &max)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside findmineig..."<<endl;
  #endif

  int loop_count;
  long int i;
  REAL norm, inorm, old_norm, delta;
  Vec3 vr_1, vr_2, vr_3;

  delta=max-mass2;

  // starting gauss vector p
  loc_p->gauss(); 
  for(i=0; i<sizeh; i++)
     {
     d_vector1[i]=(loc_p->fermion[i]).l2norm2();
     }
  global_sum(d_vector1,sizeh);
  norm=sqrt(d_vector1[0]);

  loop_count=0;
  do              // loop start
    {
    // normalize  p 
    // and r=p
    inorm=1.0/norm;
    for(i=0; i<sizeh; i++)
       {
       vr_1=(loc_p->fermion[i]);
       vr_1*=inorm;
       (loc_p->fermion[i])=vr_1;
       (loc_r->fermion[i])=vr_1;  // r=p
       }
    old_norm=norm;

    Doe(loc_h,loc_p); 
    Deo(loc_p,loc_h);   // p=DeoDoe r
    for(i=0; i<sizeh; i++)
       {
       vr_1=(loc_r->fermion[i]);
       vr_2=(loc_p->fermion[i]);
       vr_3=delta*vr_1+vr_2;
       (loc_p->fermion[i])=vr_3;   // p=max r - (M^dag M)r
       d_vector1[i]=vr_3.l2norm2();
       }
    global_sum(d_vector1,sizeh);
    norm=sqrt(d_vector1[0]); 

    old_norm=fabs(old_norm-norm);
    old_norm/=norm;
    loop_count++;
    } while(old_norm>1.0e-5);   // loop end
  
  min=max-norm;
  
  #ifdef DEBUG_MODE
  cout << "\tterminated findmineig in "<<loop_count<<" iterations"<<endl;
  #endif
  }



void findminmax(REAL &min, REAL &max)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside findminmax ..."<< endl;
  #endif

  if(use_stored==0 && update_iteration>=therm_updates)
    {
    min=min_stored;
    max=max_stored;
    }
  if((use_stored==0 && update_iteration<therm_updates) || use_stored==1)  // eigenvalues calculation in thermalization
    {                                                                     // and metropolis test
    #ifndef USE_GPU
      findmaxeig(max);
      findmineig(min, max);

      min_stored=min;
      max_stored=max;
    #else
      loc_p->gauss();
      smartpack_fermion(simple_fermion_packed, loc_p);   

      REAL min_l, max_l;

      cuda_find_max(&max_l);
      cuda_find_min(&min_l, max_l);

      min_stored=min_l;
      min=min_l;
      max_stored=max_l;
      max=max_l;
    #endif
    }

  #ifdef PARAMETER_TEST
  cout << "\nCondition number = " << max/min << "\n";
  cout << "Optimal number of pseudofermions = " << 0.5*log(max/min) << "\n"<<endl; // see Clark,Kennedy Phys.Rev.Lett. 98(2007) 051601 hep-lat/0608015
  #endif

  #ifdef DEBUG_MODE
  cout << "\tterminated findminmax [ min="<<min<<"  "<<"max="<<max<<" ]"<< endl;
  #endif
  }

