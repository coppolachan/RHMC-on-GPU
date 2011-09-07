#ifndef VEC3_H_
#define VEC3_H_

#include <complex>
#include "include/global_const.h"

// Not very good practice classes are too coupled together
class Su3;
class Fermion;
class MultiFermion;
class ShiftMultiFermion;
/////////////////////////

using namespace std;

class Vec3 {
private:
 complex<REAL> comp[3];
public:
 Vec3(void);
 Vec3(complex<REAL> input[3]);

 Vec3 operator~(void) const;
 Vec3& operator=(const Vec3 &rhs);
 void operator+=(const Vec3 &rhs);
 void operator-=(const Vec3 &rhs);
 void operator*=(float rhs);
 void operator*=(double rhs);
 void operator*=(complex<REAL> rhs);

 void zero(void);
 void unitarize(void);
 REAL l2norm(void);
 REAL l2norm2(void);

 // defined Gauss/gauss.cc
 void gauss(void);
 void z2noise(void);

 friend ostream& operator<<(ostream &os, const Vec3 &M);
 friend istream& operator>>(istream &is, Vec3 &M);
 friend Vec3 operator+(Vec3 lhs, Vec3 rhs);
 friend Vec3 operator-(Vec3 lhs, Vec3 rhs);
 friend Vec3 operator*(Vec3 lhs, float p);
 friend Vec3 operator*(float p, Vec3 rhs);
 friend Vec3 operator*(Vec3 lhs, double p);
 friend Vec3 operator*(double p, Vec3 rhs);
 friend Vec3 operator*(Vec3 lhs, complex<REAL> p);
 friend Vec3 operator*(complex<REAL> p, Vec3 rhs);
 friend complex<REAL> c_scalprod(Vec3 lhs, Vec3 rhs);
 friend REAL r_scalprod(Vec3 lhs, Vec3 rhs);

 // defined in Su3/su3.cc
 friend Vec3 operator*(const Su3 &mat, const Vec3 &vec);
 friend Su3 operator^(const Vec3 &lhs, const Vec3 &rhs);

 // defined in Packer/packer.cc
 friend void smartpack_fermion(float out[6*sizeh], const Fermion *in);
 friend void smartpack_fermion_d(float out[6*sizeh*2], const Fermion *in);
 friend void smartpack_multifermion(float *out , const MultiFermion *in);
 friend void smartunpack_multishiftfermion(ShiftMultiFermion *out, const float *in,  int order);

 };



// Vec3 class MEMBER FUNCTIONS
// base constructor
inline Vec3::Vec3(void)
  {
  comp[0]=complex<REAL>(0.0,0.0);
  comp[1]=complex<REAL>(0.0,0.0);
  comp[2]=complex<REAL>(0.0,0.0);
  }


// constructo from input
inline Vec3::Vec3(complex<REAL> input[3])
  {
  comp[0]=input[0];
  comp[1]=input[1];
  comp[2]=input[2];
  }


// complex conjugate
inline Vec3 Vec3::operator~(void) const
  {
  register Vec3 aux;
  aux.comp[0]=conj(comp[0]); 
  aux.comp[1]=conj(comp[1]); 
  aux.comp[2]=conj(comp[2]); 

  return aux;
  }


// assignement operator
inline Vec3& Vec3::operator=(const Vec3 &rhs)
  {
  comp[0]=rhs.comp[0];
  comp[1]=rhs.comp[1];
  comp[2]=rhs.comp[2];

  return *this;
  }


// operator +=
inline void Vec3::operator+=(const Vec3 &rhs)
  {
  comp[0]+=rhs.comp[0];
  comp[1]+=rhs.comp[1];
  comp[2]+=rhs.comp[2];
  }


// operator -=
inline void Vec3::operator-=(const Vec3 &rhs)
  {
  comp[0]-=rhs.comp[0];
  comp[1]-=rhs.comp[1];
  comp[2]-=rhs.comp[2];
  }


// operator *= with float
inline void Vec3::operator*=(float p)
  {
  comp[0]*=p;
  comp[1]*=p;
  comp[2]*=p;
  }


// operator *= with double
inline void Vec3::operator*=(double p)
  {
  comp[0]*=p;
  comp[1]*=p;
  comp[2]*=p;
  }


// operator *= with complex
inline void Vec3::operator*=(complex<REAL> p)
  {
  comp[0]*=p;
  comp[1]*=p;
  comp[2]*=p;
  }


// zero vector
inline void Vec3::zero(void)
  {
  comp[0]=complex<REAL>(0.0,0.0);
  comp[1]=complex<REAL>(0.0,0.0);
  comp[2]=complex<REAL>(0.0,0.0);
  }


// operator +
inline Vec3 operator+(Vec3 lhs, Vec3 rhs)
  {
  lhs+=rhs;
  return lhs;
  }


// operator -
inline Vec3 operator-(Vec3 lhs, Vec3 rhs)
  {
  lhs-=rhs;
  return lhs;
  }


// operator * with a float
inline Vec3 operator*(Vec3 lhs, float p)
  {
  lhs*=p;
  return lhs;
  }


// operator * with a float
inline Vec3 operator*(float p, Vec3 rhs)
  {
  rhs*=p;
  return rhs;
  }


// operator * with a double
inline Vec3 operator*(Vec3 lhs, double p)
  {
  lhs*=p;
  return lhs;
  }


// operator * with a double
inline Vec3 operator*(double p, Vec3 rhs)
  {
  rhs*=p;
  return rhs;
  }


// operator * with a complex
inline Vec3 operator*(Vec3 lhs, complex<REAL> p)
  {
  lhs*=p;
  return lhs;
  }


// operator * with a complex
inline Vec3 operator*(complex<REAL> p, Vec3 rhs)
  {
  rhs*=p;
  return rhs;
  }


// scalar product of two vectors
inline complex<REAL> c_scalprod(Vec3 lhs, Vec3 rhs)
  {
  complex<REAL> ris=complex<REAL>(0.0,0.0);

  ris+=conj(lhs.comp[0])*rhs.comp[0];
  ris+=conj(lhs.comp[1])*rhs.comp[1];
  ris+=conj(lhs.comp[2])*rhs.comp[2];
  
  return ris;
  }


// scalar product of two vectors, real part
inline REAL r_scalprod(Vec3 lhs, Vec3 rhs)
  {
  complex<REAL> ris=complex<REAL>(0.0,0.0);

  ris+=conj(lhs.comp[0])*rhs.comp[0];
  ris+=conj(lhs.comp[1])*rhs.comp[1];
  ris+=conj(lhs.comp[2])*rhs.comp[2];
  
  return real(ris);
  }



#endif //VEC3_H_
