#include <iostream>
#include "include/configuration.h"
#include "include/global_var.h"
#include "include/tools.h"

// plaquette measurement 
void Conf::calc_plaq(REAL &pls, REAL &plt) const
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside Conf::calc_plaq ..."<<endl;
  #endif

  Su3 aux;
  long int r, index_mu, index_nu;
  int mu, nu;
  double pls_loc, plt_loc;

  for(r=0; r<size; r++)
     {
     d_vector1[r]=0.0;
     d_vector2[r]=0.0;
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

           aux=  u_save[index_mu + r];          // 1
           aux*= u_save[index_nu + nnp[r][mu]]; // 2
           aux*=~u_save[index_mu + nnp[r][nu]]; // 3
           aux*=~u_save[index_nu + r];

           if(nu==3)
             {
             d_vector1[r]-=(double)aux.retr();  // temporal
             }
           else                                 // MINUS SIGN DUE TO STAGGERED PHASES
             {
             d_vector2[r]-=(double)aux.retr();  // spatial
             }
           }
        }
     }
  global_sum(d_vector1, size);
  global_sum(d_vector2, size);

  plt_loc=d_vector1[0]*inv_size_by_three*one_by_three;
  pls_loc=d_vector2[0]*inv_size_by_three*one_by_three;   // 3 plaquettes for site and 3 colors

  plt=plt_loc;
  pls=pls_loc;

  #ifdef DEBUG_MODE
  cout << "\tterminated Conf::calc_plaq"<<endl;
  #endif
  }



// polyakov loop measurement
void Conf::calc_poly(REAL &re, REAL &im) const 
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside Conf::calc_poly ..."<<endl;
  #endif

  long int r, r2, t;
  double re_loc, im_loc;
  Su3 aux;

  for(r=0; r<vol3; r++)   // r<vol3 iff t=0
     {
     aux.one();
     r2=r;
     for(t=0; t<nt; t++)
        {
        aux*=u_save[r2+size3];
        r2=nnp[r2][3];
        }

     d_vector1[r]=-(double)aux.retr();    // MINUS SIGN DUE TO ANTIPERIODIC TEMPORAL B.C. FOR STAGGERED PHASES
     d_vector2[r]=-(double)aux.imtr();
     }
  
  global_sum(d_vector1, vol3);
  global_sum(d_vector2, vol3);

  re_loc=d_vector1[0]*inv_vol3*one_by_three;
  im_loc=d_vector2[0]*inv_vol3*one_by_three;   // three colors

  re=re_loc;
  im=im_loc;

  #ifdef DEBUG_MODE
  cout << "\tterminated Conf::calc_poly"<<endl;
  #endif
  }


