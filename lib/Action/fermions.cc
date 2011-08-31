#include <iostream>
#include "lib/Action/fermions.h"
#include "include/tools.h"

// base constructor
Fermion::Fermion(void)
 {
 for(long int i=0; i<sizeh; i++)
    {
    fermion[i].zero();
    }
 }

// Initialize with random gaussian complex numbers
void Fermion::gauss(void)
 {
 for(long int i=0; i<sizeh; i++)
    {
    fermion[i].gauss();
    }
 }

// Initialize with z2 noise
void Fermion::z2noise(void)
 {
 for(long int i=0; i<sizeh; i++)
    {
    fermion[i].z2noise();
    }
 }

// squared L2 norm of the fermion
double Fermion::l2norm2(void)
 {
 for(long int i=0; i<sizeh; i++)
    {
    d_vector1[i]=(fermion[i].l2norm2());
    }
 global_sum(d_vector1, sizeh);

 return d_vector1[0];
 }

 //-----------------------------------------------------

// base construction
ShiftFermion::ShiftFermion(void)
 {
 for(int i=0; i<max_approx_order; i++)
    {
    for(long int j=0; j<sizeh; j++)
       {
       fermion[i][j].zero();
       }
    }
 }


//-----------------------------------------------------

// base construction
MultiFermion::MultiFermion(void)
 {
 for(int i=0; i<no_ps; i++)
    {
    for(long int j=0; j<sizeh; j++)
       {
       fermion[i][j].zero();
       }
    }
 }

// Initialize fermion_phi with gaussian random numbers
void create_phi(void)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside create_phi ..."<<endl;
 #endif

 for(int i=0; i<no_ps; i++)
    {
    for(long int j=0; j<sizeh; j++)
       {
       (fermion_phi->fermion[i][j]).gauss();
       }
    }
 #ifdef DEBUG_MODE
 cout << "\tterminated create_phi"<<endl;
 #endif
 }


// Extract the i-th fermion from the MultiFermion "in" 
void extract_fermion(Fermion *out, MultiFermion *in, int i)
 {
 for(long int j=0; j<sizeh; j++)
    {
    out->fermion[j]=in->fermion[i][j];
    }
 }


//---------------------------------------------------


// base construction
ShiftMultiFermion::ShiftMultiFermion(void)
 {
 for(int i=0; i<no_ps; i++)
    {
    for(int k=0; k<max_approx_order; k++)
       {
       for(long int j=0; j<sizeh; j++)
          {
          fermion[i][k][j].zero();
          }
       }
    }
 }

// extract the i-th pseudfermion from the ShiftMultiFermion "in"
void extract_fermion(ShiftFermion *out, ShiftMultiFermion *in, int i)
 {
 int j;
 long int l;

 for(j=0; j<max_approx_order; j++)
    {
    for(l=0; l>sizeh; l++)
       {
       out->fermion[j][l]=in->fermion[i][j][l];
       }
    } 
 }

