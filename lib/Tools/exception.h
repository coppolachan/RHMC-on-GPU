#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <exception>
#include "include/global_macro.h"

using namespace std;

void test_param(void);

class myexception0: public exception
{
  virtual const char* what() const throw()
  {
    return "\033[31mERROR: configuration file '" QUOTEME(CONF_FILE) "' does not exists\033[0m\n";
  }
} static no_stored_conf;

class myexception1: public exception
{
  virtual const char* what() const throw()
  {
    return "\033[31mERROR: stored configuration has wrong dimensions\033[0m\n";
  }
} static stored_conf_not_fit;

class myexception2: public exception
{
  virtual const char* what() const throw()
  {
    return "\033[31mERROR: multistep integrator cannot be used in PURE_GAUGE simulations\033[0m\n";
  }
} static no_multistep;

class myexception3: public exception
{
  virtual const char* what() const throw()
  {
    return "\033[31mERROR: nx, ny, nz, nt have to be even to use odd/even preconditioning\033[0m\n";
  }
} static odd_sides;

class myexception4: public exception
{
  virtual const char* what() const throw()
  {
  return "\033[31mERROR: volume must be divisible by 2*NUM_THREADS\033[0m\n";
  }
} static wrong_num_threads;




#endif //EXCEPTION_H_
