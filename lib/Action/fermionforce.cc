#include <iostream>
#include "include/global_var.h" 

#ifdef USE_GPU
#include"lib/Cuda/cuda_fermion_force.h"
#include"include/cu_inverter.h"
#include"lib/Tools/Packer/packer.h"
#endif

void fermionforce(void)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside fermionforce ..."<<endl;
  #endif
  complex<REAL> eim, emim;

  if (GlobalChemPotPar::Instance().UseChem()) {
    eim=complex<REAL>(GlobalChemPotPar::Instance().getEim_cos(), 
		      GlobalChemPotPar::Instance().getEim_sin());
    emim=complex<REAL>(GlobalChemPotPar::Instance().getEim_cos(),
		       -GlobalChemPotPar::Instance().getEim_sin());
  }
  
  int pseudofermion, iter, mu;
  long int even, odd, x, y;
  RationalApprox approx;
  Vec3 vr_1, vr_2;
  Su3 aux;

  // approximation of x^{-beta}    beta=no_vlavours/4/no_ps
  approx.md_inv_approx_coeff();

  // shift inverter
  #ifdef USE_GPU
    int order;
    get_order(order, approx);

    cu_multips_shifted_invert(GlobalParams::Instance().getResidueMD(), approx);
  #else
    multips_shifted_invert(fermion_shiftmulti, fermion_chi, GlobalParams::Instance().getResidueMD(), approx);
  #endif

  // force reconstruction
  #ifdef USE_GPU 
    float num[max_approx_order];
    get_numerators(num, approx);
    cuda_fermion_force(order, num);  // here fermion force is also multiplied by u_work and taken TA
  #else
    for(pseudofermion=0; pseudofermion<no_ps; pseudofermion++)
       {
       for(iter=0; iter<(approx.approx_order); iter++)
          {
          for(even=0; even<sizeh; even++)
             {
             (loc_s->fermion[even])=(fermion_shiftmulti->fermion[pseudofermion][iter][even]);
             }

          Doe(loc_h, loc_s);
  
          for(even=0; even<sizeh; even++)
             {
             odd=even+sizeh;
             for(mu=0; mu<3; mu++)
                {
                x=even;
                y=nnp[even][mu]-sizeh;     // sizeh<=nnp[even][mu]<size
                vr_1=(loc_h->fermion[y]);
                vr_2=~(loc_s->fermion[x]);
                vr_1*=(approx.RA_a[iter]);
                aux=(vr_1^vr_2);
                (gauge_ipdot->ipdot[mu*size+even])+=aux;

                y=nnp[odd][mu];
                vr_1=(loc_s->fermion[y]);
                vr_2=~(loc_h->fermion[x]);
                vr_1*=(-approx.RA_a[iter]);
                aux=(vr_1^vr_2);
                (gauge_ipdot->ipdot[mu*size+odd])+=aux;
                }
             for(mu=3; mu<4; mu++)
                {
                x=even;
                y=nnp[even][mu]-sizeh;     // sizeh<=nnp[even][mu]<size
                vr_1=(loc_h->fermion[y]);
                vr_2=~(loc_s->fermion[x]);
                #ifdef IM_CHEM_POT
                  vr_1*=(approx.RA_a[iter]*eim);
                #else
                  vr_1*=(approx.RA_a[iter]);
                #endif
                aux=(vr_1^vr_2);
                (gauge_ipdot->ipdot[mu*size+even])+=aux;

                y=nnp[odd][mu];
                vr_1=(loc_s->fermion[y]);
                vr_2=~(loc_h->fermion[x]);
                #ifdef IM_CHEM_POT
                  vr_1*=(-approx.RA_a[iter]*eim);
                #else
                  vr_1*=(-approx.RA_a[iter]);
                #endif
                aux=(vr_1^vr_2);
                (gauge_ipdot->ipdot[mu*size+odd])+=aux;
                }
             } 
          }
       }
  #endif

  #ifdef DEBUG_MODE
  cout << "\tterminated fermionforce"<<endl;
  #endif
 }
