#include "lib/Action/su3.h"

#include "include/tools.h"

// Su3 class MEMBER FUNCTIONS



// random SU(3) matrix assignement
void Su3::rand_matrix(void)
  {
  int i, j;
  complex<REAL> prod;
  REAL norm;
  Su3 ris;

  for(i=0; i<2; i++)
     {
     for(j=0; j<3; j++)
        {
        ris.comp[i][j]=complex<REAL>(1.0-2.0*uniform_generator(), 1.0-2.0*uniform_generator());
        }
     }
 
  norm=0.0;
  for(i=0; i<3; i++)
     {
     norm+=real(ris.comp[0][i])*real(ris.comp[0][i])+imag(ris.comp[0][i])*imag(ris.comp[0][i]);
     }
  norm=1.0/sqrt(norm);
  for(i=0; i<3; i++)
     {
     ris.comp[0][i]*=norm;
     }
  
  prod=complex<REAL>(0.0,0.0);
  for(i=0; i<3; i++)
     {
     prod+=conj(ris.comp[0][i])*ris.comp[1][i];
     }
  for(i=0; i<3; i++)
     {
     ris.comp[1][i]-=prod*ris.comp[0][i];
     }
  norm=0.0;
  for(i=0; i<3; i++)
     {
     norm+=real(ris.comp[1][i])*real(ris.comp[1][i])+imag(ris.comp[1][i])*imag(ris.comp[1][i]);
     }
  norm=1.0/sqrt(norm);
  for(i=0; i<3; i++)
     {
     ris.comp[1][i]*=norm;
     }

  prod=ris.comp[0][1]*ris.comp[1][2]-ris.comp[0][2]*ris.comp[1][1];
  ris.comp[2][0]=conj(prod);
  prod=ris.comp[0][2]*ris.comp[1][0]-ris.comp[0][0]*ris.comp[1][2];
  ris.comp[2][1]=conj(prod);
  prod=ris.comp[0][0]*ris.comp[1][1]-ris.comp[0][1]*ris.comp[1][0];
  ris.comp[2][2]=conj(prod);

  *this=ris;
  }



// project on SU(3)
void Su3::sunitarize(void)
  {
  int i;
  complex<REAL> prod;
  REAL norm;
 
  norm=0.0;
  for(i=0; i<3; i++)
     {
     norm+=real(comp[0][i])*real(comp[0][i])+imag(comp[0][i])*imag(comp[0][i]);
     }
  norm=1.0/sqrt(norm);
  for(i=0; i<3; i++)
     {
     comp[0][i]*=norm;
     }
  
  prod=complex<REAL>(0.0,0.0);
  for(i=0; i<3; i++)
     {
     prod+=conj(comp[0][i])*comp[1][i];
     }
  for(i=0; i<3; i++)
     {
     comp[1][i]-=prod*comp[0][i];
     }
  norm=0.0;
  for(i=0; i<3; i++)
     {
     norm+=real(comp[1][i])*real(comp[1][i])+imag(comp[1][i])*imag(comp[1][i]);
     }
  norm=1.0/sqrt(norm);
  for(i=0; i<3; i++)
     {
     comp[1][i]*=norm;
     }

  prod=comp[0][1]*comp[1][2]-comp[0][2]*comp[1][1];
  comp[2][0]=conj(prod);
  prod=comp[0][2]*comp[1][0]-comp[0][0]*comp[1][2];
  comp[2][1]=conj(prod);
  prod=comp[0][0]*comp[1][1]-comp[0][1]*comp[1][0];
  comp[2][2]=conj(prod);
  }


// L2 norm of the matrix
REAL Su3::l2norm(void) const 
  {
  REAL aux;

  aux=0.0;
  aux+=real(comp[0][0])*real(comp[0][0])+imag(comp[0][0])*imag(comp[0][0]);
  aux+=real(comp[0][1])*real(comp[0][1])+imag(comp[0][1])*imag(comp[0][1]);
  aux+=real(comp[0][2])*real(comp[0][2])+imag(comp[0][2])*imag(comp[0][2]);

  aux+=real(comp[1][0])*real(comp[1][0])+imag(comp[1][0])*imag(comp[1][0]);
  aux+=real(comp[1][1])*real(comp[1][1])+imag(comp[1][1])*imag(comp[1][1]);
  aux+=real(comp[1][2])*real(comp[1][2])+imag(comp[1][2])*imag(comp[1][2]);

  aux+=real(comp[2][0])*real(comp[2][0])+imag(comp[2][0])*imag(comp[2][0]);
  aux+=real(comp[2][1])*real(comp[2][1])+imag(comp[2][1])*imag(comp[2][1]);
  aux+=real(comp[2][2])*real(comp[2][2])+imag(comp[2][2])*imag(comp[2][2]);

  return sqrt(aux);
  }


// L2 norm squared of the matrix
REAL Su3::l2norm2(void) const 
  {
  REAL aux;

  aux=0.0;
  aux+=real(comp[0][0])*real(comp[0][0])+imag(comp[0][0])*imag(comp[0][0]);
  aux+=real(comp[0][1])*real(comp[0][1])+imag(comp[0][1])*imag(comp[0][1]);
  aux+=real(comp[0][2])*real(comp[0][2])+imag(comp[0][2])*imag(comp[0][2]);

  aux+=real(comp[1][0])*real(comp[1][0])+imag(comp[1][0])*imag(comp[1][0]);
  aux+=real(comp[1][1])*real(comp[1][1])+imag(comp[1][1])*imag(comp[1][1]);
  aux+=real(comp[1][2])*real(comp[1][2])+imag(comp[1][2])*imag(comp[1][2]);

  aux+=real(comp[2][0])*real(comp[2][0])+imag(comp[2][0])*imag(comp[2][0]);
  aux+=real(comp[2][1])*real(comp[2][1])+imag(comp[2][1])*imag(comp[2][1]);
  aux+=real(comp[2][2])*real(comp[2][2])+imag(comp[2][2])*imag(comp[2][2]);

  return aux;
  }



// traceless anti-hermitian part
void Su3::ta(void)
  {
  int i, j;
  complex<REAL> trace=complex<REAL>(0.0,0.0);
  Su3 aux;

  aux=*this;

  for(i=0; i<3; i++)
     {
     for(j=0; j<3; j++)
        {
        comp[i][j]=(aux.comp[i][j]-conj(aux.comp[j][i]))*complex<REAL>(0.5,0.0);
        }
     trace+=comp[i][i];
     }

  trace*=one_by_three;
  for(i=0; i<3; i++)
     {
     comp[i][i]-=trace;
     }
  }


// exponential
void Su3::exp(void)
  {
  Su3 aux, ris;

  ris.one();
  aux=*this;

  ris+=aux;  // ris=1+x
  
  aux*=*this;
  aux*=0.5;
  ris+=aux;  // 2 order

  aux*=*this;
  aux*=one_by_three;
  ris+=aux;  // 3 order

  aux*=*this;
  aux*=0.25;
  ris+=aux;  // 4 order

  aux*=*this;
  aux*=0.2;
  ris+=aux;  // 5 order

  aux*=*this;
  aux*=0.166666666666666666;
  ris+=aux;  // 6 order

  ris.sunitarize();

  *this=ris;
  }


					// FRIEND FUNCTIONS

// print a matrix on ostream
ostream& operator<<(ostream &os, const Su3 &M)
  {
  os << M.comp[0][0] << " " << M.comp[0][1] << " " << M.comp[0][2] <<"\n";
  os << M.comp[1][0] << " " << M.comp[1][1] << " " << M.comp[1][2] <<"\n";
  os << M.comp[2][0] << " " << M.comp[2][1] << " " << M.comp[2][2] <<"\n\n";

  return os;
  }


// copy matrix from istream
istream& operator>>(istream &is, Su3 &M)
  {
  is >> M.comp[0][0] >> M.comp[0][1] >> M.comp[0][2];
  is >> M.comp[1][0] >> M.comp[1][1] >> M.comp[1][2];
  is >> M.comp[2][0] >> M.comp[2][1] >> M.comp[2][2];
  
  return is;
  }

// product between matrices
Su3 operator*(Su3 lhs, Su3 rhs)
  {
  lhs*=rhs;
  return lhs;
  }

