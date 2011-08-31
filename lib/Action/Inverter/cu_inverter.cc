// shifted inverter for multiple pseudofermions: for each pseudofermion component of "in" 
// solve the equation {(M^dag M) + RA_b[i]}out=in   
// Jegerlehner hep-lat/9612014 "Krylov space solvers for shifted linear systems


#ifdef USE_GPU
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "include/global_var.h"
#include "include/tools.h"



#include"lib/Tools/Packer/packer.h"
#include"lib/Cuda/cuda_inverter.h"




void cu_multips_shifted_invert (REAL res, RationalApprox approx)
  {
  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  cout <<"DEBUG: inside cu_multips_shifted_invert ..."<<endl;

  cout << approx;
  #endif

  #ifdef TIMING_CUDA_CPP
  clock_t time_start, time_finish;
  time_start=clock();
  #endif

  long int i;
  int iter, cg, cg_aux[no_ps], pseudofermion, order;
  double shifts[max_approx_order];

  get_order(order, approx);
  get_shifts(shifts, approx);

  // start loop on pseudofermions
  for(pseudofermion=0; pseudofermion<no_ps; pseudofermion++)
     {

     if(res<inv_single_double_prec)
       {
       cuda_shifted_inverter_d(res,
                      shifts,
                      order,
                      pseudofermion, 
                      &cg);
       }
     else
       {
       cuda_shifted_inverter(res,
                             shifts,
                             order,
                             pseudofermion, 
                             &cg);
       }

     if(cg==max_cg)
       {
       ofstream err_file;
       err_file.open(QUOTEME(ERROR_FILE), ios::app);   
       err_file  << "WARNING: maximum number of iterations reached in cuda_multips_shift_invert\n";
       err_file.close();
       exit(1);
       }

     cg_aux[pseudofermion]=cg;
     } // end loop on pseudofermions

  #ifdef TIMING_CUDA_CPP
  time_finish=clock();
  cout << "time for cu_multips_shifted_invert = " << ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
  #endif

  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  cout << "\tterminated cu_multips_shifted_invert ( stop_res=" <<res<<" )"<<endl;
  for(i=0; i<no_ps; i++)
     {
     cout << "\t\tcg[pseudofermion n. "<< i <<"]="<< cg_aux[i]<<endl;
     }

  // test
  MultiFermion *fermion_aux;
  fermion_aux=new MultiFermion;
  Vec3 vr_1, vr_2, vr_3, vr_4, vr_5;

  smartunpack_multishiftfermion(fermion_shiftmulti, psi_packed, order);
  smartunpack_gauge(gauge_conf, gauge_field_packed);
  smartunpack_multifermion(fermion_aux, chi_packed);

  for(pseudofermion=0; pseudofermion<no_ps; pseudofermion++)
     {
     for(iter=0; iter<order; iter++)
        {
        for(i=0; i<sizeh; i++)
           {
           (loc_p->fermion[i])=(fermion_shiftmulti->fermion[pseudofermion][iter][i]);
           }

        Doe(loc_h, loc_p);
        Deo(loc_s, loc_h);
        for(i=0; i<sizeh; i++)
           {
           vr_1=(loc_p->fermion[i]);
           vr_2=(loc_s->fermion[i]);
           vr_3=(fermion_aux->fermion[pseudofermion][i]);
           vr_4=(mass2+shifts[iter])*vr_1 -vr_2 -vr_3; // (M^dagM+RA_b)out-in
           d_vector1[i]=vr_4.l2norm2();
           }

        global_sum(d_vector1, sizeh);
        cout << "\t\t[ pseudoferm="<<pseudofermion<<" iter="<<iter<< " res/stop_res="<< sqrt(d_vector1[0])/res << " ]"<<endl;
        }
     }
  delete fermion_aux;
  #endif
  }

#endif
