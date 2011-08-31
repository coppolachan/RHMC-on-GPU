#ifndef SU3_H_
#define SU3_H_

#include <complex>
#include "lib/Tools/vec3.h"

// Not very good practice
class Conf;
class Ipdot;
class Momenta;
////////////////////////

using namespace std;

class Su3 {
private:
 complex<REAL> comp[3][3];
public:
 Su3(void);
 Su3(complex<REAL> input[3][3]);

 Su3 operator~(void) const; 
 Su3& operator=(const Su3 &rhs);
 void operator+=(const Su3 &rhs);
 void operator-=(const Su3 &rhs);
 void operator*=(int rhs);
 void operator*=(float rhs);
 void operator*=(double rhs);
 void operator*=(complex<REAL> rhs);
 void operator*=(const Su3 &rhs);

 void one(void);
 void zero(void);
 void rand_matrix(void);

 REAL retr(void) const;
 REAL imtr(void) const;
 complex<REAL> tr(void) const;
 complex<REAL> det(void) const;
 void sunitarize(void);
 REAL l2norm(void) const;
 REAL l2norm2(void) const;
 void ta(void);
 void exp(void);
 void row_multiply(int i, int p);
 void row_multiply(int i, float p);
 void row_multiply(int i, double p);
 void row_multiply(int i, complex<REAL> p);

 // defined in Gauss/gauss.cc
 void gauss(void);

 friend ostream& operator<<(ostream &os, const Su3 &M);
 friend istream& operator>>(istream &is, Su3 &M);
 friend Su3 operator+(Su3 lhs, Su3 rhs);
 friend Su3 operator-(Su3 lhs, Su3 rhs);
 friend Su3 operator*(Su3 lhs, float p);
 friend Su3 operator*(float p, Su3 rhs);
 friend Su3 operator*(Su3 lhs, double p);
 friend Su3 operator*(double p, Su3 rhs);
 friend Su3 operator*(Su3 lhs, complex<REAL> p);
 friend Su3 operator*(complex<REAL> p, Su3 rhs);
 friend Su3 operator*(Su3 lhs, Su3 rhs);

 friend Vec3 operator*(const Su3 &mat, const Vec3 &vec);
 friend Su3 operator^(const Vec3 &lhs, const Vec3 &rhs);

 // defined in Packer/cupacker.cc
 friend void smartpack_gauge(float out[2*12*no_links] , const Conf *in);
 friend void smartpack_tamatrix(float out[8*no_links], Ipdot *in);
 friend void smartpack_thmatrix(float out[8*no_links], Momenta *in);
 };
 
// base constructor
inline Su3::Su3()
  {
  comp[0][0]=complex<REAL>(0.0,0.0);
  comp[0][1]=complex<REAL>(0.0,0.0);
  comp[0][2]=complex<REAL>(0.0,0.0);

  comp[1][0]=complex<REAL>(0.0,0.0);
  comp[1][1]=complex<REAL>(0.0,0.0);
  comp[1][2]=complex<REAL>(0.0,0.0);

  comp[2][0]=complex<REAL>(0.0,0.0);
  comp[2][1]=complex<REAL>(0.0,0.0);
  comp[2][2]=complex<REAL>(0.0,0.0);
  }


// costructor from imput
inline Su3::Su3(complex<REAL> input[3][3])
  {
  comp[0][0]=input[0][0];
  comp[0][1]=input[0][1];
  comp[0][2]=input[0][2];

  comp[1][0]=input[1][0];
  comp[1][1]=input[1][1];
  comp[1][2]=input[1][2];

  comp[2][0]=input[2][0];
  comp[2][1]=input[2][1];
  comp[2][2]=input[2][2];
  }


// hermitian conjugate
inline Su3 Su3::operator~(void) const
  {
  register Su3 ris;

  ris.comp[0][0]=conj(comp[0][0]);
  ris.comp[0][1]=conj(comp[1][0]);
  ris.comp[0][2]=conj(comp[2][0]);

  ris.comp[1][0]=conj(comp[0][1]);
  ris.comp[1][1]=conj(comp[1][1]);
  ris.comp[1][2]=conj(comp[2][1]);

  ris.comp[2][0]=conj(comp[0][2]);
  ris.comp[2][1]=conj(comp[1][2]);
  ris.comp[2][2]=conj(comp[2][2]);

  return ris;
  }


// assignement operator
inline Su3& Su3::operator=(const Su3 &rhs)
  {
  comp[0][0]=rhs.comp[0][0];
  comp[0][1]=rhs.comp[0][1];
  comp[0][2]=rhs.comp[0][2];

  comp[1][0]=rhs.comp[1][0];
  comp[1][1]=rhs.comp[1][1];
  comp[1][2]=rhs.comp[1][2];

  comp[2][0]=rhs.comp[2][0];
  comp[2][1]=rhs.comp[2][1];
  comp[2][2]=rhs.comp[2][2];

  return *this;
  }


// operator += between matrices
inline void Su3::operator+=(const Su3 &rhs)
  {
  comp[0][0]+=rhs.comp[0][0];
  comp[0][1]+=rhs.comp[0][1];
  comp[0][2]+=rhs.comp[0][2];

  comp[1][0]+=rhs.comp[1][0];
  comp[1][1]+=rhs.comp[1][1];
  comp[1][2]+=rhs.comp[1][2];

  comp[2][0]+=rhs.comp[2][0];
  comp[2][1]+=rhs.comp[2][1];
  comp[2][2]+=rhs.comp[2][2];
  }


// operator -= between matrices
inline void Su3::operator-=(const Su3 &rhs)
  {
  comp[0][0]-=rhs.comp[0][0];
  comp[0][1]-=rhs.comp[0][1];
  comp[0][2]-=rhs.comp[0][2];

  comp[1][0]-=rhs.comp[1][0];
  comp[1][1]-=rhs.comp[1][1];
  comp[1][2]-=rhs.comp[1][2];

  comp[2][0]-=rhs.comp[2][0];
  comp[2][1]-=rhs.comp[2][1];
  comp[2][2]-=rhs.comp[2][2];
  }


// operator *= with a int
inline void Su3::operator*=(int rhs)
  {
  comp[0][0]*=rhs;
  comp[0][1]*=rhs;
  comp[0][2]*=rhs;

  comp[1][0]*=rhs;
  comp[1][1]*=rhs;
  comp[1][2]*=rhs;

  comp[2][0]*=rhs;
  comp[2][1]*=rhs;
  comp[2][2]*=rhs;
  }


// operator *= with a float
inline void Su3::operator*=(float rhs)
  {
  comp[0][0]*=rhs;
  comp[0][1]*=rhs;
  comp[0][2]*=rhs;

  comp[1][0]*=rhs;
  comp[1][1]*=rhs;
  comp[1][2]*=rhs;

  comp[2][0]*=rhs;
  comp[2][1]*=rhs;
  comp[2][2]*=rhs;
  }


// operator *= with a double
inline void Su3::operator*=(double rhs)
  {
  comp[0][0]*=rhs;
  comp[0][1]*=rhs;
  comp[0][2]*=rhs;

  comp[1][0]*=rhs;
  comp[1][1]*=rhs;
  comp[1][2]*=rhs;

  comp[2][0]*=rhs;
  comp[2][1]*=rhs;
  comp[2][2]*=rhs;
  }


// operator *= with a complex
inline void Su3::operator*=(complex<REAL> rhs)
  {
  comp[0][0]*=rhs;
  comp[0][1]*=rhs;
  comp[0][2]*=rhs;

  comp[1][0]*=rhs;
  comp[1][1]*=rhs;
  comp[1][2]*=rhs;

  comp[2][0]*=rhs;
  comp[2][1]*=rhs;
  comp[2][2]*=rhs;
  }


// operator *= with a matrix
inline void Su3::operator*=(const Su3 &rhs)
  {
  register Su3 aux;

  aux=*this;

  register complex<REAL>p0=rhs.comp[0][0];
  register complex<REAL>p1=rhs.comp[1][0];
  register complex<REAL>p2=rhs.comp[2][0];

  comp[0][0]=p0*aux.comp[0][0];
  comp[1][0]=p0*aux.comp[1][0];
  comp[2][0]=p0*aux.comp[2][0];

  comp[0][0]+=p1*aux.comp[0][1];
  comp[1][0]+=p1*aux.comp[1][1];
  comp[2][0]+=p1*aux.comp[2][1];

  comp[0][0]+=p2*aux.comp[0][2];
  comp[1][0]+=p2*aux.comp[1][2];
  comp[2][0]+=p2*aux.comp[2][2];

  p0=rhs.comp[0][1];
  p1=rhs.comp[1][1];
  p2=rhs.comp[2][1];

  comp[0][1]=p0*aux.comp[0][0];
  comp[1][1]=p0*aux.comp[1][0];
  comp[2][1]=p0*aux.comp[2][0];

  comp[0][1]+=p1*aux.comp[0][1];
  comp[1][1]+=p1*aux.comp[1][1];
  comp[2][1]+=p1*aux.comp[2][1];

  comp[0][1]+=p2*aux.comp[0][2];
  comp[1][1]+=p2*aux.comp[1][2];
  comp[2][1]+=p2*aux.comp[2][2];

  p0=rhs.comp[0][2];
  p1=rhs.comp[1][2];
  p2=rhs.comp[2][2];

  comp[0][2]=p0*aux.comp[0][0];
  comp[1][2]=p0*aux.comp[1][0];
  comp[2][2]=p0*aux.comp[2][0];

  comp[0][2]+=p1*aux.comp[0][1];
  comp[1][2]+=p1*aux.comp[1][1];
  comp[2][2]+=p1*aux.comp[2][1];

  comp[0][2]+=p2*aux.comp[0][2];
  comp[1][2]+=p2*aux.comp[1][2];
  comp[2][2]+=p2*aux.comp[2][2];
  }


// zero operator
inline void Su3::zero(void)
  {
  comp[0][0]=complex<REAL>(0.0,0.0);
  comp[0][1]=complex<REAL>(0.0,0.0);
  comp[0][2]=complex<REAL>(0.0,0.0);

  comp[1][0]=complex<REAL>(0.0,0.0);
  comp[1][1]=complex<REAL>(0.0,0.0);
  comp[1][2]=complex<REAL>(0.0,0.0);

  comp[2][0]=complex<REAL>(0.0,0.0);
  comp[2][1]=complex<REAL>(0.0,0.0);
  comp[2][2]=complex<REAL>(0.0,0.0);
  }


// identity operator
inline void Su3::one(void)
  {
  comp[0][0]=complex<REAL>(1.0,0.0);
  comp[0][1]=complex<REAL>(0.0,0.0);
  comp[0][2]=complex<REAL>(0.0,0.0);

  comp[1][0]=complex<REAL>(0.0,0.0);
  comp[1][1]=complex<REAL>(1.0,0.0);
  comp[1][2]=complex<REAL>(0.0,0.0);

  comp[2][0]=complex<REAL>(0.0,0.0);
  comp[2][1]=complex<REAL>(0.0,0.0);
  comp[2][2]=complex<REAL>(1.0,0.0);
  }


// real part of trace
inline REAL Su3::retr(void) const 
  {
  complex<REAL> aux;

  aux=comp[0][0]+comp[1][1]+comp[2][2];

  return real(aux);
  }
  

// immaginary part of trace
inline REAL Su3::imtr(void) const
  {
  complex<REAL> aux;

  aux=comp[0][0]+comp[1][1]+comp[2][2];

  return imag(aux);
  }


// trace
inline complex<REAL> Su3::tr(void) const
  {
  complex<REAL> aux;

  aux=comp[0][0]+comp[1][1]+comp[2][2];

  return aux;
  }


// determinant
inline complex<REAL> Su3::det(void) const
  {
  complex<REAL> aux;

  aux=comp[0][0]*comp[1][1]*comp[2][2]+comp[1][0]*comp[2][1]*comp[0][2]+comp[2][0]*comp[0][1]*comp[1][2];
  aux-=comp[2][0]*comp[1][1]*comp[0][2]+comp[1][0]*comp[0][1]*comp[2][2]+comp[0][0]*comp[2][1]*comp[1][2];

  return aux;
  }


// multiply i-th row by p
inline void Su3::row_multiply(int i, int p)
 {
 comp[i][0]*=p;
 comp[i][1]*=p;
 comp[i][2]*=p;
 }


// multiply i-th row by p
inline void Su3::row_multiply(int i, float p)
 {
 comp[i][0]*=p;
 comp[i][1]*=p;
 comp[i][2]*=p;
 }


// multiply i-th row by p
inline void Su3::row_multiply(int i, double p)
 {
 comp[i][0]*=p;
 comp[i][1]*=p;
 comp[i][2]*=p;
 }


// multiply i-th row by p
inline void Su3::row_multiply(int i, complex<REAL> p)
 {
 comp[i][0]*=p;
 comp[i][1]*=p;
 comp[i][2]*=p;
 }




// sum of matrices
inline Su3 operator+(Su3 lhs, Su3 rhs)
  {
  lhs+=rhs;
  return lhs;
  }


// difference of matrices
inline Su3 operator-(Su3 lhs, Su3 rhs)
  {
  lhs-=rhs;
  return lhs;
  }


// product with float
inline Su3 operator*(Su3 lhs, float p)
  {
  lhs*=p;
  return lhs;
  }


// product with float
inline Su3 operator*(float p, Su3 rhs)
  {
  rhs*=p;
  return rhs;
  }


// product with double
inline Su3 operator*(Su3 lhs, double p)
  {
  lhs*=p;
  return lhs;
  }


// product with double
inline Su3 operator*(double p, Su3 rhs)
  {
  rhs*=p;
  return rhs;
  }


// product with complex
inline Su3 operator*(Su3 lhs, complex<REAL> p)
  {
  lhs*=p;
  return lhs;
  }


// product with complex
inline Su3 operator*(complex<REAL> p, Su3 rhs)
  {
  rhs*=p;
  return rhs;
  }


// matrix * vector product
inline Vec3 operator*(const Su3 &mat, const Vec3 &vec)
  {
  register Vec3 ris;
  register complex<REAL> p0=vec.comp[0];
  register complex<REAL> p1=vec.comp[1];
  register complex<REAL> p2=vec.comp[2];
 
  ris.comp[0]=p0*mat.comp[0][0];
  ris.comp[1]=p0*mat.comp[1][0];
  ris.comp[2]=p0*mat.comp[2][0];

  ris.comp[0]+=p1*mat.comp[0][1];
  ris.comp[1]+=p1*mat.comp[1][1];
  ris.comp[2]+=p1*mat.comp[2][1];

  ris.comp[0]+=p2*mat.comp[0][2];
  ris.comp[1]+=p2*mat.comp[1][2];
  ris.comp[2]+=p2*mat.comp[2][2];
 
  return ris;
  }


// direct product of vectors
inline Su3 operator^(const Vec3 &lhs, const Vec3 &rhs)
  {
  Su3 ris;

  ris.comp[0][0]=lhs.comp[0]*rhs.comp[0];
  ris.comp[0][1]=lhs.comp[0]*rhs.comp[1];
  ris.comp[0][2]=lhs.comp[0]*rhs.comp[2];

  ris.comp[1][0]=lhs.comp[1]*rhs.comp[0];
  ris.comp[1][1]=lhs.comp[1]*rhs.comp[1];
  ris.comp[1][2]=lhs.comp[1]*rhs.comp[2];

  ris.comp[2][0]=lhs.comp[2]*rhs.comp[0];
  ris.comp[2][1]=lhs.comp[2]*rhs.comp[1];
  ris.comp[2][2]=lhs.comp[2]*rhs.comp[2];

  return ris;
  }



#endif //SU3_H_
