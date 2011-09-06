// use z2 noise instead of gaussian noise (see hep-lat/9308015)
// use the global defined fermions loc_chi, loc_phi, rnd_o, rnd_e, chi_o and loc_h

#include <iostream>
#include <fstream>
#include <complex>
#include "include/tools.h"

#ifdef USE_GPU
#include"lib/Tools/Packer/packer.h"
#include"lib/Cuda/cuda_inverter.h"
#include"lib/Cuda/cuda_init.h"
#endif

using namespace std;

void chiral_meas(complex<REAL> &chiral, complex<REAL> &e_dens, complex<REAL> &b_dens, complex<REAL> &p_dens)
  {
  #ifdef DEBUG
  cout << "DEBUG: inside chiral_meas ..."<<endl;
  #endif

  #ifdef IM_CHEM_POT
  const complex<REAL> eim=complex<REAL>(eim_cos, eim_sin);
  const complex<REAL> emim=complex<REAL>(eim_cos, -eim_sin);
  #endif

  complex<double> *c_vector1, *c_vector2;
  c_vector1=new complex<double>[size];  // complex auxiliary vectors to be used in energy and baryon 
  c_vector2=new complex<double>[size];  // density when imaginary chemical potential is used

  if(GlobalParams::Instance().getRandVect()>0)
    {
    int k, iter;
    long int i, j, index1, index2;
    complex<double> loc_chiral, loc_e_dens, loc_b_dens, loc_p_dens;
    complex<double> p1, p2;
    Vec3 v_e, v_o, w_e, w_o;

    loc_chiral=complex<double>(0.0,0.0);
    loc_e_dens=complex<double>(0.0,0.0);
    loc_b_dens=complex<double>(0.0,0.0);
    loc_p_dens=complex<double>(0.0,0.0);
 
    for(iter=0; iter<GlobalParams::Instance().getRandVect(); iter++)
       {
       rnd_e->z2noise();
       rnd_o->z2noise();

       Deo(phi_e, rnd_o);
       for(i=0; i<sizeh; i++)
          {
	    (phi_e->fermion[i])=GlobalParams::Instance().getMass()*
	      (rnd_e->fermion[i])-(phi_e->fermion[i]);
          }

       #ifndef USE_GPU
          invert(chi_e, phi_e, GlobalParams::Instance().getResidueMetro());
       #else
         smartpack_fermion_d(simple_fermion_packed, phi_e);
         cuda_meas_init();

         int cg, order;
         double *shifts = new double[max_approx_order];
         int psferm=0;
         get_order(order, *meas_inv_coeff);  // order=1
         get_shifts(shifts, *meas_inv_coeff); // just one shift equal to 0.0

         cuda_shifted_inverter_d(GlobalParams::Instance().getResidueMetro(), shifts, order, psferm, &cg);
         if(cg==GlobalParams::Instance().getMaxCG())
           {
           ofstream err_file;
           err_file.open(QUOTEME(ERROR_FILE), ios::app);   
           err_file  << "WARNING: maximum number of iterations reached in cuda_multips_shift_invert in chiralmeas\n";
           err_file.close();
           }
         delete [] shifts;

         cuda_meas_end();
         smartunpack_fermion_d(chi_e, simple_fermion_packed);
       #endif
   
       Doe(phi_o, chi_e);
       for(i=0; i<sizeh; i++)
          { 
	    chi_o->fermion[i]=GlobalParams::Instance().getMassInv()*
	      (rnd_o->fermion[i] - phi_o->fermion[i]);
          } 

       // chiral condensate calculation
       for(i=0; i<sizeh; i++)
          { 
          c_vector1[i]=c_scalprod(rnd_o->fermion[i], chi_o->fermion[i]); 
          c_vector1[i]+=c_scalprod(rnd_e->fermion[i], chi_e->fermion[i]);
          }
       global_sum(c_vector1,sizeh);
       loc_chiral+=c_vector1[0]*
	 complex<double>(GlobalParams::Instance().getNf()*0.25*inv_size,0.0); 
   
       #ifndef IM_CHEM_POT
         // energy and baryon density calculation
         for(i=0; i<sizeh; i++)   // i=even index
            {
            j=i+sizeh;            // j=odd
            index1=i+size3;
            index2=nnp[i][3]-sizeh;    // nnp[even,3]=odd   sizeh<=odd<size

            v_e=(gauge_conf->u_work[index1])*(chi_o->fermion[index2]);
            w_e=(gauge_conf->u_work[index1])*(rnd_o->fermion[index2]);
 
            p1=c_scalprod(rnd_e->fermion[i], v_e);
            p2=c_scalprod(w_e, chi_e->fermion[i]);

            c_vector1[i]=p1-p2;
            c_vector2[i]=p1+p2;

            index1=j+size3;
            index2=nnp[j][3];

            v_o=(gauge_conf->u_work[index1])*(chi_e->fermion[index2]);
            w_o=(gauge_conf->u_work[index1])*(rnd_e->fermion[index2]);

            p1=c_scalprod(rnd_o->fermion[i], v_o);
            p2=c_scalprod(w_o, chi_o->fermion[i]);

            c_vector1[i]+=p1-p2;
            c_vector2[i]+=p1+p2;
            }
         global_sum(c_vector1,sizeh);
         loc_e_dens+=c_vector1[0]*complex<double>(0.5*GlobalParams::Instance().getNf()*0.25*inv_size,0.0);  // energy density
         global_sum(c_vector2,sizeh);
         loc_b_dens+=c_vector2[0]*complex<double>(0.5*GlobalParams::Instance().getNf()*0.25*inv_size,0.0);  // barion density
       #else
         // energy and baryon density calculation
         for(i=0; i<sizeh; i++)   // i=even index
            {
            j=i+sizeh;            // j=odd
            index1=i+size3;
            index2=nnp[i][3]-sizeh;    // nnp[even,3]=odd   sizeh<=odd<size

            v_e=(gauge_conf->u_work[index1])*(chi_o->fermion[index2]);
            w_e=(gauge_conf->u_work[index1])*(rnd_o->fermion[index2]);
 
            c_vector1[i]=c_scalprod(rnd_e->fermion[i], v_e);
            c_vector2[i]=c_scalprod(w_e, chi_e->fermion[i]);

            index1=j+size3;
            index2=nnp[j][3];

            v_o=(gauge_conf->u_work[index1])*(chi_e->fermion[index2]);
            w_o=(gauge_conf->u_work[index1])*(rnd_e->fermion[index2]);

            c_vector1[i]+=c_scalprod(rnd_o->fermion[i], v_o);
            c_vector2[i]+=c_scalprod(w_o, chi_o->fermion[i]);
            }
         global_sum(c_vector1,sizeh);
         global_sum(c_vector2,sizeh);

         loc_e_dens+=(eim*c_vector1[0]-emim*c_vector2[0])*complex<double>(0.5*GlobalParams::Instance().getNf()*0.25*inv_size,0.0);  // energy density
         loc_b_dens+=(eim*c_vector1[0]+emim*c_vector2[0])*complex<double>(0.5*GlobalParams::Instance().getNf()*0.25*inv_size,0.0);  // barion density
       #endif

       // pressure density calculation
       for(i=0; i<sizeh; i++)    // i=even
          {
          j=i+sizeh;             // j=odd
          c_vector1[i]=0.0;
          for(k=0; k<3; k++)
             {
             index1=i+k*size;
             index2=nnp[i][k]-sizeh;
  
             v_e=(gauge_conf->u_work[index1])*(chi_o->fermion[index2]);
             w_e=(gauge_conf->u_work[index1])*(rnd_o->fermion[index2]);
 
             c_vector1[i]+=c_scalprod(rnd_e->fermion[i], v_e);
             c_vector1[i]-=c_scalprod(w_e, chi_e->fermion[i]);

             index1=j+k*size;
             index2=nnp[j][k];
 
             v_o=(gauge_conf->u_work[index1])*(chi_e->fermion[index2]);
             w_o=(gauge_conf->u_work[index1])*(rnd_e->fermion[index2]);

             c_vector1[i]+=c_scalprod(rnd_o->fermion[i], v_o);
             c_vector1[i]-=c_scalprod(w_o, chi_o->fermion[i]);
             }
          }
       global_sum(c_vector1,sizeh);
       loc_p_dens+=c_vector1[0]*complex<double>(0.5*GlobalParams::Instance().getNf()*0.25*inv_size,0.0);     // pressure density
       }

    p1=complex<double>(1.0/(double) GlobalParams::Instance().getRandVect(), 0.0);

    chiral=loc_chiral*p1;
    e_dens=loc_e_dens*p1;
    b_dens=loc_b_dens*p1;
    p_dens=loc_p_dens*p1;
    }

  delete [] c_vector1;
  delete [] c_vector2;

  #ifdef DEBUG_MODE
  cout << "\tterminated chiral_meas"<<endl;
  #endif
  }


void meas(ofstream &out)
  {
  int i;
  REAL ps, pt, pr, pi;
  complex<REAL> chiral_cond, energy_density, barion_density, pressure_density;

  gauge_conf->calc_plaq(ps,pt);
  gauge_conf->calc_poly(pr,pi);

  for(i=0; i<4; i++)
     {
     out << update_iteration << "  ";
     //   plaqs        plaqt
     out << ps << "  " << pt << "  ";

     //  poly_re       poly_im 
     out << pr << "  " << pi << "  ";

     chiral_meas(chiral_cond, energy_density, barion_density, pressure_density);
     out << real(chiral_cond) << "  " << imag(chiral_cond) << "  ";
     out << real(energy_density) << "  " << imag(energy_density) << "  ";
     out << real(barion_density) << "  " << imag(barion_density) << "  ";
     out << real(pressure_density) << "  " << imag(pressure_density) << endl;
     }

  }
