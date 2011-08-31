#ifndef STAPLES_H_
#define STAPLES_H_

#include "lib/Action/su3.h"

class Staples {
private:
  Su3 staples[no_links];
public:
  Staples(void);
 
  friend void calc_staples(void);

  // defined in Ipdot/ipdot.cc
  friend void calc_ipdot_gauge(void);
  friend void calc_ipdot(void);
};

void calc_staples(void);

#endif //STAPLES_H_
