// Calculate "out" from  (M^dag M) * out = in
// using the CG-algorithm
// (M^dag M) = mass2 - Deo*Doe
// Using: loc_r, loc_h, loc_s, loc_p   

#include <iostream>
#include <fstream>
#include "lib/Action/fermions.h"
#include "include/tools.h"


void invert (Fermion *out, Fermion *in, REAL res)
  {
  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  cout << "DEBUG: inside invert ..."<< endl;
  #endif

  int cg;
  long int i;
  double delta, alpha, lambda, omega, gammag;
  Vec3 vr_1, vr_2, vr_3, vr_4, vr_5;

  // initializations
  out->gauss();      // trial solution for out

  // s=(M^dagM)out
  Doe(loc_h,out);
  Deo(loc_s, loc_h);
  for(i=0; i<sizeh; i++)
     {
     vr_1=(out->fermion[i]);
     vr_2=(loc_s->fermion[i]);
     loc_s->fermion[i]=mass2*vr_1-vr_2;
     }

  // r=in-s, p=r, delta=(r,r)
  for(i=0; i<sizeh; i++)
     {
     vr_3=(in->fermion[i])-(loc_s->fermion[i]);

     loc_r->fermion[i]=vr_3;
     loc_p->fermion[i]=vr_3;
     d_vector1[i]=vr_3.l2norm2();
     }
  global_sum(d_vector1, sizeh);
  delta=d_vector1[0];

  // loop over cg iterations
  cg=0;
  do {
     cg++;
    
     // s=(M^dag M)p    alpha=(p,s)
     Doe(loc_h, loc_p);
     Deo(loc_s, loc_h);
  
     for(i=0; i<sizeh; i++)
        {
        vr_1=(loc_p->fermion[i]);
        vr_2=(loc_s->fermion[i]);
        vr_3=mass2*vr_1 - vr_2;
        (loc_s->fermion[i])=vr_3; 
        d_vector1[i]=r_scalprod(vr_1,vr_3);
        }
     global_sum(d_vector1, sizeh);
     alpha=d_vector1[0];

     omega=delta/alpha;
     
     // out+=omega*p  r-=omega*s
     // lambda=(r,r);
     for(i=0; i<sizeh; i++)
        {
        vr_1=(out->fermion[i]);
        vr_2=(loc_p->fermion[i]);
        vr_3=vr_1 + omega*vr_2;       
        (out->fermion[i])=vr_3;

        vr_3=(loc_r->fermion[i]);
        vr_4=(loc_s->fermion[i]);
        vr_5=vr_3 - omega*vr_4;
        (loc_r->fermion[i])=vr_5;

        d_vector1[i]=vr_5.l2norm2();
        } 
     global_sum(d_vector1, sizeh);
     lambda=d_vector1[0];

     gammag=lambda/delta;
     delta=lambda;

     // p=r+gammag*p
     for(i=0; i<sizeh; i++)
        {
        vr_1=(loc_r->fermion[i]);
        vr_2=(loc_p->fermion[i]);
        vr_3=vr_1+gammag*vr_2;
        (loc_p->fermion[i])=vr_3;
        }

     } while( (sqrt(lambda)>res) && cg<max_cg);
  
  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  cout << "\tterminated invert terminated in "<<cg<<" iterations [";
  // test
  Doe(loc_h, out);
  Deo(loc_s, loc_h);
  for(i=0; i<sizeh; i++)
     {
     vr_1=(out->fermion[i]);
     vr_2=(loc_s->fermion[i]);
     vr_3=(in->fermion[i]);
     vr_4=mass2*vr_1 -vr_2 -vr_3;
     d_vector1[i]=vr_4.l2norm2();
     }
  global_sum(d_vector1, sizeh);
  cout << " res/stop_res="<< sqrt(d_vector1[0])/res << " ,  stop_res="<< res << " ]"<<endl;
  #endif

  if(cg==max_cg)
    {
    ofstream err_file;
    err_file.open(QUOTEME(ERROR_FILE), ios::app);   
    err_file << "WARNING: maximum number of iterations reached in invert\n";
    err_file.close();
    }
  }



// shifted inverter for multiple pseudofermions: for each pseudofermion component of "in" 
// solve the equation {(M^dag M)+RA_b[i]}out=in   
// Jegerlehner hep-lat/9612014 "Krylov space solvers for shifted linear systems
//
// USES the global defined ShiftFermion p_shiftferm and the global defined Fermion loc_h, loc_s, loc_p, loc_r
// 
// attention to the order of the indeces!!
// Fermion->fermion[sizeh]
// ShiftFermion->fermion[max_approx_order][sizeh]
// MultiFermion->fermion[no_ps][sizeh]
// ShiftMultiFermion->fermion[no_ps][max_approx_order][sizeh]
void multips_shifted_invert (ShiftMultiFermion *out, MultiFermion *in, REAL res, RationalApprox approx)
  {
  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  cout <<"DEBUG: inside multips_shifted_invert ..."<<endl;

  cout << approx;
  #endif

  #ifdef TIMING_CUDA_CPP
  clock_t time_start, time_finish;
  time_start=clock();
  #endif

  long int i;
  int iter, cg, cg_aux[no_ps];
  int flag[max_approx_order], pseudofermion;
  double alpha, delta, lambda, omega, omega_save, gammag, fact;
  REAL zeta_i[max_approx_order], zeta_ii[max_approx_order], zeta_iii[max_approx_order];
  REAL omegas[max_approx_order], gammas[max_approx_order];
  Vec3 vr_1, vr_2, vr_3, vr_4, vr_5;
 
  // start loop on pseudofermions
  for(pseudofermion=0; pseudofermion<no_ps; pseudofermion++)
     {
     // trial solution out = 0, set all flag to 1
     for(iter=0; iter<(approx.approx_order); iter++)
        {
        for(i=0; i<sizeh; i++)
           {
           out->fermion[pseudofermion][iter][i].zero();
           }

        flag[iter]=1;
        }

     // r=in, p=phi delta=(r,r)
     for(i=0; i<sizeh; i++)
        {
        vr_3=in->fermion[pseudofermion][i];
        (loc_r->fermion[i])=vr_3;
        (loc_p->fermion[i])=vr_3;
        d_vector1[i]=vr_3.l2norm2();
        }
     global_sum(d_vector1, sizeh);
     delta=d_vector1[0];

     omega=1.0;
  
     for(iter=0; iter<(approx.approx_order); iter++)
        {
        for(i=0; i<sizeh; i++)
           {
           (p_shiftferm->fermion[iter][i])=(in->fermion[pseudofermion][i]);   // ps_0=phi
           } 
        zeta_i[iter]=1.0;         // zeta_{-1}=1.0
        zeta_ii[iter]=1.0;        // zeta_{ 0}=1.0
        gammas[iter]=0.0;         // gammas_{-1}=0.0
        }    
 
     gammag=0.0;
     cg=0;
     do {      // loop over cg iterations
       cg++;
 
       // s=(M^dagM)p, alhpa=(p,s)=(p,Ap)
       Doe(loc_h, loc_p);
       Deo(loc_s, loc_h);
       for(i=0; i<sizeh; i++)
          { 
          vr_1=(loc_p->fermion[i]);
          vr_2=(loc_s->fermion[i]);
          vr_3=mass2*vr_1-vr_2;
          (loc_s->fermion[i])=vr_3;
          d_vector1[i]=r_scalprod(vr_1,vr_3);
          }
       global_sum(d_vector1,sizeh);
       alpha=d_vector1[0]; 

       omega_save=omega;   // omega_save=omega_(j-1)
       omega=-delta/alpha;  // omega = (r_j,r_j)/(p_j, Ap_j)
   
       // out-=omegas*ps 
       for(iter=0; iter<(approx.approx_order); iter++)
          {
          if(flag[iter]==1)
            {
            zeta_iii[iter] = (zeta_i[iter]*zeta_ii[iter]*omega_save)/
                ( omega*gammag*(zeta_i[iter]-zeta_ii[iter])+
                        zeta_i[iter]*omega_save*(1.0-(approx.RA_b[iter])*omega) );
            omegas[iter]=omega*zeta_iii[iter]/zeta_ii[iter];
            for(i=0; i<sizeh; i++)
               {
               vr_1=(out->fermion[pseudofermion][iter][i]);
               vr_2=(p_shiftferm->fermion[iter][i]);
               (out->fermion[pseudofermion][iter][i])=vr_1-omegas[iter]*vr_2;
               } 
            }
          }
       
       // r+=omega*s; lambda=(r,r)
       for(i=0; i<sizeh; i++)
          {
          vr_3 = (loc_r->fermion[i]);
          vr_4 = (loc_s->fermion[i]);
          vr_5 = vr_3+omega*vr_4;
          (loc_r->fermion[i])=vr_5;
          d_vector1[i]=vr_5.l2norm2();
          }
       global_sum(d_vector1, sizeh); 
       lambda=d_vector1[0];
   
       gammag=lambda/delta;

       // p=r+gammag*p
       for(i=0; i<sizeh; i++)
          {
          vr_1=(loc_r->fermion[i]);
          vr_2=(loc_p->fermion[i]);
          (loc_p->fermion[i])=vr_1+gammag*vr_2;
          }

       // ps=r+gammag*ps 
       for(iter=0; iter<(approx.approx_order); iter++)
          {
          if(flag[iter]==1)
            {
            gammas[iter]=gammag*zeta_iii[iter]*omegas[iter]/(zeta_ii[iter]*omega);
            for(i=0; i<sizeh; i++)
               {
               vr_1=(loc_r->fermion[i]);
               vr_3=(p_shiftferm->fermion[iter][i]);
               (p_shiftferm->fermion[iter][i])=zeta_iii[iter]*vr_1+gammas[iter]*vr_3;
               }
            fact=sqrt(delta*zeta_ii[iter]*zeta_ii[iter]);
            if(fact<res)
              {
              flag[iter]=0;
              }

            zeta_i[iter]=zeta_ii[iter];
            zeta_ii[iter]=zeta_iii[iter];
            }
          }
       delta=lambda;

       } while(sqrt(lambda)>res && cg<max_cg); // end of cg iterations 

     if(cg==max_cg)
       {
       ofstream err_file;
       err_file.open(QUOTEME(ERROR_FILE), ios::app);   
       err_file << "WARNING: maximum number of iterations reached in multips_shifted_invert\n";
       err_file.close();
       }
    
     cg_aux[pseudofermion]=cg;

     } // end loop on pseudofermions

  #ifdef TIMING_CUDA_CPP
  time_finish=clock();
  cout << "time for multips_shifted_invert = " << ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
  #endif

  #if ((defined DEBUG_MODE) || (defined DEBUG_INVERTER))
  cout << "\tterminated multips_shift_invert ( stop_res=" <<res<<" )"<<endl;
  for(i=0; i<no_ps; i++)
     {
     cout << "\t\tcg[pseudofermion n. "<< i <<"]="<< cg_aux[i]<<endl;
     }

  // test
  for(pseudofermion=0; pseudofermion<no_ps; pseudofermion++)
     {
     for(iter=0; iter<(approx.approx_order); iter++)
        {
        for(i=0; i<sizeh; i++)
           {
           (loc_p->fermion[i])=(out->fermion[pseudofermion][iter][i]);
           }

        Doe(loc_h, loc_p);
        Deo(loc_s, loc_h);
        for(i=0; i<sizeh; i++)
           {
           vr_1=(loc_p->fermion[i]);
           vr_2=(loc_s->fermion[i]);
           vr_3=(in->fermion[pseudofermion][i]);
           vr_4=(mass2+approx.RA_b[iter])*vr_1 -vr_2 -vr_3; // (M^dagM+RA_b)out-in
           d_vector1[i]=vr_4.l2norm2();
           }

       global_sum(d_vector1, sizeh);
       cout << "\t\t[ pseudoferm="<<pseudofermion<<" iter="<<iter<< " res/stop_res="<< sqrt(d_vector1[0])/res << " ]"<<endl;
        }
     }
  #endif
  }
