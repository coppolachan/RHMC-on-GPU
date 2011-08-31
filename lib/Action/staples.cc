#include <iostream>
#include "include/global_var.h"
#include "lib/Action/staples.h"

// base constructor
Staples::Staples(void)
 {
 for(long int i=0; i<no_links; i++)
    {
    staples[i].zero();
    }
 }


// calculate staples 
void calc_staples(void)
 {
 #ifdef DEBUG_MODE
 cout <<"DEBUG: inside calc_staples ..."<<endl;
 #endif
 int mu, nu;
 long int pos, index_mu, index_nu, helper;
 Su3 aux;

 for(pos=0; pos < size; pos++)
    {
    for(mu=0; mu<4; mu++)
       {
       index_mu=mu*size;

       (gauge_staples->staples[index_mu+pos]).zero();
      
       for(nu=0; nu<4; nu++)
          {
          if(nu!=mu)
            {
            index_nu=nu*size;

            //             (1)
            //           +-->--+
            //           |     |
            // ^mu       |     V  (2)   to calculate (1)*(2)*(3)
            // |         |     |
            //           +--<--+
            // ->nu   pos  (3)
               
            aux = (gauge_conf->u_work[index_nu + nnp[pos][mu]]);  // 1
            aux*=~(gauge_conf->u_work[index_mu + nnp[pos][nu]]);  // 2 
            aux*=~(gauge_conf->u_work[index_nu + pos]);           // 3

            (gauge_staples->staples[index_mu+pos])+=aux;

            //             (1)
            //           +--<--+
            //           |     |
            // ^mu   (2) V     |   to calculate (1)*(2)*(3)
            // |         |     |
            //           +-->--+
            // ->nu  help  (3)  temp
   
            helper=nnm[pos][nu];
            aux =~(gauge_conf->u_work[index_nu + nnp[helper][mu]]); // 1
            aux*=~(gauge_conf->u_work[index_mu + helper]);          // 2
            aux*= (gauge_conf->u_work[index_nu + helper]);          // 3

            (gauge_staples->staples[index_mu+pos])+=aux;
            }
          }
       }
    }
 #ifdef DEBUG_MODE
 cout <<"\tterminated calc_staples"<<endl;
 #endif
 }
