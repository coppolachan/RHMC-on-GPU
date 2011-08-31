#include <iostream>
#include "include/configuration.h"
#include "include/global_var.h"

// gauge part of acton
void Conf::loc_action(double *value, int init) const
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside loc_action ..."<<endl;
  #endif

  Su3 aux;
  long int r, index_mu, index_nu;
  int mu, nu;
  double p;

  for(r=0; r<size; r++)
     {
     // if inti==0 initialize action to 0
     if(init==0)
       {
       value[r]=0.0; 
       }

     for(mu=0; mu<3; mu++) 
        {
        index_mu=mu*size; 
        for(nu=mu+1; nu<4; nu++)
           {
           //            (3)
           //          +<---+
           // nu       |    ^ 
           // ^    (4) V    | (2)
           // |        +--->+
           // |        r (1)
           // +---> mu
 
           index_nu=nu*size;

           aux=  (gauge_conf->u_work[index_mu + r]);          // 1
           aux*= (gauge_conf->u_work[index_nu + nnp[r][mu]]); // 2
           aux*=~(gauge_conf->u_work[index_mu + nnp[r][nu]]); // 3
           aux*=~(gauge_conf->u_work[index_nu + r]);          // 4
 
           p=+beta_by_three*aux.retr();   // PLUS SIGN DUE TO STAGGERED PHASES
       
           if(init==0)
             {    
             value[r]+=p;
             }
           if(init==-1)
             { 
             value[r]-=p;
             }
           }
        }
     }
  #ifdef DEBUG_MODE
  cout << "\tterminated loc_action"<<endl;
  #endif
  }


// fermion part of action
void fermion_action(double *value, int init)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside fermion_action ..."<<endl;
  #endif

  if(init==0)       // action=<fermion_phi, fermion_phi>
    {
    long int i;
    int ps;
    Vec3 vr_1;

    for(ps=0; ps<no_ps; ps++)
       {
       for(i=0; i<sizeh; i++)
          {
          vr_1=(fermion_phi->fermion[ps][i]);
          value[i]+=r_scalprod(vr_1, vr_1);
          }
       }
    }

  if(init==-1)      // action=<fermion_chi, fermion_phi>
    {
    long int i;
    int ps;
    Vec3 vr_1, vr_2;
  
    for(ps=0; ps<no_ps; ps++)
       {
       for(i=0; i<sizeh; i++)
          {
          vr_1=(fermion_phi->fermion[ps][i]);
          vr_2=(fermion_chi->fermion[ps][i]);
          value[i]-=r_scalprod(vr_1, vr_2);
          }
       }
    }

  #ifdef DEBUG_MODE
  cout << "\tterminated fermion_action"<<endl;
  #endif
  }


// total action
void calc_action(double *value, int init)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside calc_action ..."<<endl;
  #endif

  long int r, i;
  Su3 aux;

  gauge_conf->loc_action(value, init);

  // momenta part of action
  if(init==0)
    {
    for(r=0; r<size; r++)
       {
       for(i=0;i<4;i++)
          {
          aux=(gauge_momenta->momenta[r+i*size]);
          aux*=(gauge_momenta->momenta[r+i*size]);
          value[r]+=0.5*aux.retr();
          }
       }
     }

  if(init==-1)
    {
    for(r=0; r<size; r++)
       {
       for(i=0;i<4;i++)
          {
          aux=(gauge_momenta->momenta[r+i*size]);
          aux*=(gauge_momenta->momenta[r+i*size]);
          value[r]-=0.5*aux.retr();
          }
       }
     }

  #ifndef PURE_GAUGE
  fermion_action(value, init);
  #endif

  #ifdef DEBUG_MODE
  cout << "\tterminated calc_action"<<endl;
  #endif
  }
