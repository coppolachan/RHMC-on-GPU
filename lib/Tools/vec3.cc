#include "lib/Tools/vec3.h"

// Warning: inline functions should be placed only in the .h files

// unitarize
void Vec3::unitarize(void)
  {
  REAL norm=0.0;

  norm+=real(comp[0])*real(comp[0])+imag(comp[0])*imag(comp[0]);
  norm+=real(comp[1])*real(comp[1])+imag(comp[1])*imag(comp[1]);
  norm+=real(comp[2])*real(comp[2])+imag(comp[2])*imag(comp[2]);

  norm=1.0/sqrt(norm);
  comp[0]*=norm;
  comp[1]*=norm;
  comp[2]*=norm;
  }


// return L2 norm 
REAL Vec3::l2norm(void)
  {
  REAL norm=0.0;

  norm+=real(comp[0])*real(comp[0])+imag(comp[0])*imag(comp[0]);
  norm+=real(comp[1])*real(comp[1])+imag(comp[1])*imag(comp[1]);
  norm+=real(comp[2])*real(comp[2])+imag(comp[2])*imag(comp[2]);

  return sqrt(norm);
  }


// return L2 norm squared 
REAL Vec3::l2norm2(void)
  {
  REAL norm2=0.0;

  norm2+=real(comp[0])*real(comp[0])+imag(comp[0])*imag(comp[0]);
  norm2+=real(comp[1])*real(comp[1])+imag(comp[1])*imag(comp[1]);
  norm2+=real(comp[2])*real(comp[2])+imag(comp[2])*imag(comp[2]);

  return norm2;
  }


					// FRIEND FUNCTIONS
// print a vector on a stream
ostream& operator<<(ostream &os, const Vec3 &v)
  {
  os << v.comp[0] << " " << v.comp[1] << " " << v.comp[2] << "\n\n";
  return os;
  }


// copy a vector froma a stream
istream& operator>>(istream &is, Vec3 &v)
  {
  is >> v.comp[0] >> v.comp[1] >> v.comp[2];
  return is;
  }

