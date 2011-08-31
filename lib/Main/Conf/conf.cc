// Functions for handling Conf class

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include "include/configuration.h"
#include "lib/Action/su3.h"
#include "include/global_var.h"
#include "include/tools.h"
#include "lib/Tools/exception.h"

// constructor & initialization
Conf::Conf(int init)
 {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside Conf::Conf ..."<<endl;
  #endif

 long int i;

 if(init==0) // start from ordered conf
   {
   #ifdef DEBUG_MODE
   cout << "\tstart from ordered conf ..."<<endl;
   #endif
   for(i=0; i<no_links; i++)
      {
      Su3 aux;               // aux is a small perturbation
      aux.rand_matrix();
      aux*=0.01;
      u_save[i].one();
      u_save[i]+=aux;
      u_save[i].sunitarize();
      u_save[i]*=eta[i];
      u_work[i]=u_save[i];
      }
   }

 if(init==1) // start from random conf
   {
   #ifdef DEBUG_MODE
   cout << "\tstart from random conf ..."<<endl;
   #endif
   for(i=0; i<no_links; i++)
      {
      u_save[i].rand_matrix();
      u_save[i]*=eta[i];
      u_work[i]=u_save[i];
      }
   update_iteration=0;
   }

 if(init==2) // start from saved conf.
   {
   #ifdef DEBUG_MODE
   cout << "\tstart from stored conf ..."<<endl;
   #endif
   if(look_for_file(QUOTEME(CONF_FILE))!=0)
     {
     throw no_stored_conf;
     }

   int nx_l, ny_l, nz_l, nt_l;
   REAL beta_l, mass_l, no_flavours_l;
   long int r;
   ifstream file;

   file.open(QUOTEME(CONF_FILE), ios::in);

   file >> nx_l;
   file >> ny_l;
   file >> nz_l;
   file >> nt_l;
   file >> beta_l;
   file >> mass_l;
   file >> no_flavours_l;
   file >> update_iteration;

   if(nx!=nx_l || ny!=ny_l || nz!=nz_l || nt !=nt_l)
     {
     throw stored_conf_not_fit;
     }

   for(r=0; r<no_links; r++)
      {
      file >> u_save[r];

      u_work[r].sunitarize();
      u_work[r].row_multiply(2,eta[r]);

      u_work[r]=u_save[r];
      }
   file.close();
   }

 #ifdef DEBUG_MODE
 cout << "\tterminated Conf::Conf"<<endl;
 #endif
 }


// copy u_work to u_save
void Conf::save(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside Conf::save ..."<<endl;
 #endif
 long int r;

 for(r=0; r<no_links; r++)
    {
    u_save[r]=u_work[r];
    }
 #ifdef DEBUG_MODE
 cout << "\tterminated Conf::save"<<endl;
 #endif
 }



// copy u_save to u_work
void Conf::copy_saved(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside Conf::copy_saved ..."<<endl;
 #endif

 long int r;

 for(r=0; r<no_links; r++)
    {
    u_work[r]=u_save[r];
    }
 #ifdef DEBUG_MODE
 cout << "\tterminated Conf::copy_saved"<<endl;
 #endif
 }


// write configuration to conf_file_0 or conf_file_1 (defined in Include/global_macro.cc)
void Conf::write(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside Conf::write ..."<<endl;
 #endif

 static int conf_count=0;

 long int r;
 ofstream file;
 char conf_name[50], aux[10];

 strcpy(conf_name, QUOTEME(CONF_FILE));
 sprintf(aux, "_%d", conf_count);
 strcat(conf_name, aux);

 file.open(conf_name, ios::out);
 file.precision(16);

 file << nx << " " << ny << " " << nz << " " << nt << " " << beta << " " << mass << " " <<no_flavours << " " <<  update_iteration <<"\n";

 for(r=0; r<no_links; r++)
    {
    file << u_save[r];
    }

 file.close();

 conf_count=1-conf_count;
 
 #ifdef DEBUG_MODE
 cout << "\tterminated Conf::write"<<endl;
 #endif
 }


// write configuration to conf_file (defined in Include/global_macro.cc)
void Conf::write_last(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside Conf::write_last ..."<<endl;
 #endif

 long int r;
 ofstream file;

 file.open(QUOTEME(CONF_FILE), ios::out);
 file.precision(16);

 file << nx << " " << ny << " " << nz << " " << nt << " " << beta << " " << mass << " " <<no_flavours << " " <<  update_iteration <<"\n";

 for(r=0; r<no_links; r++)
    {
    file << u_save[r];
    }

 file.close();

 #ifdef DEBUG_MODE
 cout << "\tterminated Conf::write_last"<<endl;
 #endif
 }


// print configuration
void Conf::print(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside Conf::print ..."<<endl;
 #endif

 long int r;

 cout<<"U_SAVE\n";
 for(r=0; r<no_links; r++)
    {
    cout << u_save[r];
    }

 cout<<"U_WORK\n";
 for(r=0; r<no_links; r++)
    {
    cout << u_work[r];
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated Conf::print"<<endl;
 #endif
 }


// unitarize and restore staggered phases
void Conf::unitarize_with_eta(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside Conf::unitarize_with_eta ..."<<endl;
 #endif

 long int r;

 for(r=0; r<no_links; r++)
    {
    u_work[r].sunitarize();
    u_work[r].row_multiply(2,eta[r]);
    }
 #ifdef DEBUG_MODE
 cout << "\tterminated Conf::unitarize_with_eta"<<endl;
 #endif
 }

