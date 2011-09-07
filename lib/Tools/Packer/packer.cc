#include <iostream>
#include <stdlib.h>
#include "lib/Tools/Packer/packer.h"
#include "lib/Update/momenta.h"


// configuration packer
// 12 = 2(row)*3(complex)*2(real)
void smartpack_gauge(float out[2*12*no_links] , const Conf *in)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartpack_gauge ..."<<endl;
 #endif

 long int i, mu;
 long int offs=12*no_links;
 double aux;

 for(mu=0; mu<4; mu++)
    {
    for(i=0; i<size; i++)
       {
       // 1st float
       out[4*i              + 12*mu*size] = (float) real(in->u_work[i + mu*size].comp[0][0]);
       out[4*i + 1          + 12*mu*size] = (float) imag(in->u_work[i + mu*size].comp[0][0]);
       out[4*i + 2          + 12*mu*size] = (float) real(in->u_work[i + mu*size].comp[0][1]);
       out[4*i + 3          + 12*mu*size] = (float) imag(in->u_work[i + mu*size].comp[0][1]);
	 
       out[4*i     + 4*size + 12*mu*size] = (float) real(in->u_work[i + mu*size].comp[0][2]);
       out[4*i + 1 + 4*size + 12*mu*size] = (float) imag(in->u_work[i + mu*size].comp[0][2]);
       out[4*i + 2 + 4*size + 12*mu*size] = (float) real(in->u_work[i + mu*size].comp[1][0]);
       out[4*i + 3 + 4*size + 12*mu*size] = (float) imag(in->u_work[i + mu*size].comp[1][0]);
	 
       out[4*i     + 8*size + 12*mu*size] = (float) real(in->u_work[i + mu*size].comp[1][1]);
       out[4*i + 1 + 8*size + 12*mu*size] = (float) imag(in->u_work[i + mu*size].comp[1][1]);
       out[4*i + 2 + 8*size + 12*mu*size] = (float) real(in->u_work[i + mu*size].comp[1][2]);
       out[4*i + 3 + 8*size + 12*mu*size] = (float) imag(in->u_work[i + mu*size].comp[1][2]);

       // 2nd float
       aux = real(in->u_work[i + mu*size].comp[0][0]) - (double) out[4*i              + 12*mu*size];
       out[offs + 4*i              + 12*mu*size] = (float) aux;
       aux = imag(in->u_work[i + mu*size].comp[0][0]) - (double) out[4*i + 1          + 12*mu*size]; 
       out[offs + 4*i + 1          + 12*mu*size] = (float) aux;
       aux = real(in->u_work[i + mu*size].comp[0][1]) - (double) out[4*i + 2          + 12*mu*size];
       out[offs + 4*i + 2          + 12*mu*size] = (float) aux;
       aux = imag(in->u_work[i + mu*size].comp[0][1]) - (double) out[4*i + 3          + 12*mu*size];
       out[offs + 4*i + 3          + 12*mu*size] = (float) aux;

       aux = real(in->u_work[i + mu*size].comp[0][2]) - (double) out[4*i     + 4*size + 12*mu*size];
       out[offs + 4*i     + 4*size + 12*mu*size] = (float) aux;
       aux = imag(in->u_work[i + mu*size].comp[0][2]) - (double) out[4*i + 1 + 4*size + 12*mu*size];
       out[offs + 4*i + 1 + 4*size + 12*mu*size] = (float) aux;
       aux = real(in->u_work[i + mu*size].comp[1][0]) - (double) out[4*i + 2 + 4*size + 12*mu*size];
       out[offs + 4*i + 2 + 4*size + 12*mu*size] = (float) aux;
       aux = imag(in->u_work[i + mu*size].comp[1][0]) - (double) out[4*i + 3 + 4*size + 12*mu*size];
       out[offs + 4*i + 3 + 4*size + 12*mu*size] = (float) aux;

       aux = real(in->u_work[i + mu*size].comp[1][1]) - (double) out[4*i     + 8*size + 12*mu*size];
       out[offs + 4*i     + 8*size + 12*mu*size] = (float) aux;
       aux = imag(in->u_work[i + mu*size].comp[1][1]) - (double) out[4*i + 1 + 8*size + 12*mu*size];
       out[offs + 4*i + 1 + 8*size + 12*mu*size] = (float) aux;
       aux = real(in->u_work[i + mu*size].comp[1][2]) - (double) out[4*i + 2 + 8*size + 12*mu*size];
       out[offs + 4*i + 2 + 8*size + 12*mu*size] = (float) aux;
       aux = imag(in->u_work[i + mu*size].comp[1][2]) - (double) out[4*i + 3 + 8*size + 12*mu*size];
       out[offs + 4*i + 3 + 8*size + 12*mu*size] = (float) aux;
       }
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartpack_gauge"<<endl;
 #endif
 }


// Unpack gauge configuration
void smartunpack_gauge(Conf *out, const float in[2*12*no_links])
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartunpack_gauge ..."<<endl;
 #endif

 long int i, pos;
 int mu;
 long int offs=12*no_links;
 REAL tempR, tempI;
 complex<REAL> aux[3][3]; 

 for(mu=0; mu<4; mu++)
    {
    for(i=0; i<size;i++)
       {
       tempR  = (double) in[       4*i              + 12*mu*size];
       tempR += (double) in[offs + 4*i              + 12*mu*size];
       tempI  = (double) in[       4*i + 1          + 12*mu*size];
       tempI += (double) in[offs + 4*i + 1          + 12*mu*size];
       aux[0][0]=complex<REAL>(tempR, tempI);

       tempR  = (double) in[       4*i + 2          + 12*mu*size];
       tempR += (double) in[offs + 4*i + 2          + 12*mu*size];
       tempI  = (double) in[       4*i + 3          + 12*mu*size];
       tempI += (double) in[offs + 4*i + 3          + 12*mu*size];
       aux[0][1]=complex<REAL>(tempR, tempI);

       tempR  = (double) in[       4*i     + 4*size + 12*mu*size];
       tempR += (double) in[offs + 4*i     + 4*size + 12*mu*size];
       tempI  = (double) in[       4*i + 1 + 4*size + 12*mu*size];
       tempI += (double) in[offs + 4*i + 1 + 4*size + 12*mu*size];
       aux[0][2]=complex<REAL>(tempR, tempI);

       tempR  = (double) in[       4*i + 2 + 4*size + 12*mu*size];
       tempR += (double) in[offs + 4*i + 2 + 4*size + 12*mu*size];
       tempI  = (double) in[       4*i + 3 + 4*size + 12*mu*size];
       tempI += (double) in[offs + 4*i + 3 + 4*size + 12*mu*size];
       aux[1][0]=complex<REAL>(tempR, tempI);

       tempR  = (double) in[       4*i     + 8*size + 12*mu*size];
       tempR += (double) in[offs + 4*i     + 8*size + 12*mu*size];
       tempI  = (double) in[       4*i + 1 + 8*size + 12*mu*size];
       tempI += (double) in[offs + 4*i + 1 + 8*size + 12*mu*size];
       aux[1][1]=complex<REAL>(tempR, tempI);

       tempR  = (double) in[       4*i + 2 + 8*size + 12*mu*size];
       tempR += (double) in[offs + 4*i + 2 + 8*size + 12*mu*size];
       tempI  = (double) in[       4*i + 3 + 8*size + 12*mu*size];
       tempI += (double) in[offs + 4*i + 3 + 8*size + 12*mu*size];
       aux[1][2]=complex<REAL>(tempR, tempI);

       aux[2][0]=complex<REAL>(0.0,0.0);
       aux[2][1]=complex<REAL>(0.0,0.0);
       aux[2][2]=complex<REAL>(0.0,0.0);

       out->u_work[i+mu*size]=Su3(aux);      
       }
    }
 out->unitarize_with_eta();

 #ifdef DEBUG_MODE
 cout << "\tterminated smartunpack_gauge"<<endl;
 #endif
 }




// fermion packer
//                                                            size
//                                                           |----|        v2
// Fermion are only on even sites on CPU                               |^^^^^^^^^|
// are on even and odd sites on GPU                            e    o    e    o    e    o
//                  |v1|                      fermion_device |----|----|----|----|----|----|
//  Fermion(site_i)=|v2|                                     |_________|         |_________|
//                  |v3|                                          v1                 v3
//
//  |--v1--|=|v1(site0).re,v1(site0).im,v1(site1).re,v1(site1).im-----v1(sizeh).re, v1(sizeh).im|
//  and similarly for the other components 
 
void smartpack_fermion(float out[6*sizeh], const Fermion *in)
{
#ifdef DEBUG_MODE
  cout << "DEBUG: inside smartpack_fermion ..."<<endl;
#endif

#ifdef DEBUG_MODE_2
  float sum = 0;
#endif
  
  long int i;
  
  for(i=0; i<sizeh; i++)
    {
      out[2*i            ] = (float) real(in->fermion[i].comp[0]);
      out[2*i          +1] = (float) imag(in->fermion[i].comp[0]);
      out[2*i +   size   ] = (float) real(in->fermion[i].comp[1]);
      out[2*i +   size +1] = (float) imag(in->fermion[i].comp[1]);
      out[2*i + 2*size   ] = (float) real(in->fermion[i].comp[2]);
      out[2*i + 2*size +1] = (float) imag(in->fermion[i].comp[2]);
#ifdef DEBUG_MODE_2
      sum += out[2*i]*out[2*i] + out[2*i+1]*out[2*i+1];
#endif
    }
  
#ifdef DEBUG_MODE_2
  cout << "DEBUG_2: [smartpack_fermion] Out Norm : "<<  sum << endl; 
#endif 


 #ifdef DEBUG_MODE
 cout << "\tterminated smartpack_fermion"<<endl;
 #endif
 }


void smartpack_fermion_d(float out[6*sizeh*2], const Fermion *in)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartpack_fermion_d ..."<<endl;
 #endif

 long int i;
 long int offs=6*sizeh;
 double aux;

 for(i=0; i<sizeh; i++)
    {
    // 1st float
    out[2*i            ] = (float) real(in->fermion[i].comp[0]);
    out[2*i          +1] = (float) imag(in->fermion[i].comp[0]);
    out[2*i +   size   ] = (float) real(in->fermion[i].comp[1]);
    out[2*i +   size +1] = (float) imag(in->fermion[i].comp[1]);
    out[2*i + 2*size   ] = (float) real(in->fermion[i].comp[2]);
    out[2*i + 2*size +1] = (float) imag(in->fermion[i].comp[2]);

    //2nd float
    aux = real(in->fermion[i].comp[0]) - (double)out[2*i            ];
    out[offs + 2*i            ] = (float) aux;
    aux = imag(in->fermion[i].comp[0]) - (double)out[2*i          +1];
    out[offs + 2*i          +1] = (float) aux;
    aux = real(in->fermion[i].comp[1]) - (double)out[2*i +   size   ];
    out[offs + 2*i +   size   ] = (float) aux;
    aux = imag(in->fermion[i].comp[1]) - (double)out[2*i +   size +1];
    out[offs + 2*i +   size +1] = (float) aux;
    aux = real(in->fermion[i].comp[2]) - (double)out[2*i + 2*size   ];
    out[offs + 2*i + 2*size   ] = (float) aux;
    aux = imag(in->fermion[i].comp[2]) - (double)out[2*i + 2*size +1];
    out[offs + 2*i + 2*size +1] = (float) aux;
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartpack_fermion_d"<<endl;
 #endif
 }



void smartunpack_fermion_d(Fermion *out, const float in[6*sizeh*2])
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartunpack_fermion_d ..."<<endl;
 #endif

 long int i;
 long int offs=6*sizeh;

 double d_re, d_im;
 complex<REAL> aux[3];

 for(i=0; i<sizeh; i++)
    {
    d_re = (double) in[2*i            ];
    d_re+= (double) in[2*i             +offs];
    d_im = (double) in[2*i          +1];
    d_im+= (double) in[2*i          +1 +offs];
    aux[0]=complex<REAL>(d_re, d_im);

    d_re = (double) in[2*i +   size   ];
    d_re+= (double) in[2*i +   size    +offs];
    d_im = (double) in[2*i +   size +1];
    d_im+= (double) in[2*i +   size +1 +offs];
    aux[1]=complex<REAL>(d_re, d_im);

    d_re = (double) in[2*i + 2*size   ];
    d_re+= (double) in[2*i + 2*size    +offs];
    d_im = (double) in[2*i + 2*size +1];
    d_im+= (double) in[2*i + 2*size +1 +offs];
    aux[2]=complex<REAL>(d_re, d_im);

    out->fermion[i]=Vec3(aux);
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartunpack_fermion_d"<<endl;
 #endif
 }




// 6*sizeh*no_ps*2=6*sizeh(fermion)*no_ps*2(1double~2float)
void smartpack_multifermion(float *out , const MultiFermion *in)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartpack_multifermion ..."<<endl;
 #endif


 long int i;
 long int offs = 6*sizeh*GlobalParams::Instance().getNumPS();
 double aux;
 int ps;

 for(ps=0; ps<GlobalParams::Instance().getNumPS(); ps++)
    {
    for(i=0; i<sizeh; i++)
       {
       // 1st float
       out[3*size*ps + 2*i            ] = (float) real(in->fermion[ps][i].comp[0]);
       out[3*size*ps + 2*i          +1] = (float) imag(in->fermion[ps][i].comp[0]);
       out[3*size*ps + 2*i +   size   ] = (float) real(in->fermion[ps][i].comp[1]);
       out[3*size*ps + 2*i +   size +1] = (float) imag(in->fermion[ps][i].comp[1]);
       out[3*size*ps + 2*i + 2*size   ] = (float) real(in->fermion[ps][i].comp[2]);
       out[3*size*ps + 2*i + 2*size +1] = (float) imag(in->fermion[ps][i].comp[2]);

       // 2nd float
       aux = real(in->fermion[ps][i].comp[0]) - (double) out[3*size*ps + 2*i            ];
       out[offs + 3*size*ps + 2*i            ] = (float) aux;
       aux = imag(in->fermion[ps][i].comp[0]) - (double) out[3*size*ps + 2*i          +1];
       out[offs + 3*size*ps + 2*i          +1] = (float) aux;
       aux = real(in->fermion[ps][i].comp[1]) - (double) out[3*size*ps + 2*i +   size   ];
       out[offs + 3*size*ps + 2*i +   size   ] = (float) aux;
       aux = imag(in->fermion[ps][i].comp[1]) - (double) out[3*size*ps + 2*i +   size +1];
       out[offs + 3*size*ps + 2*i +   size +1] = (float) aux;
       aux = real(in->fermion[ps][i].comp[2]) - (double) out[3*size*ps + 2*i + 2*size   ];
       out[offs + 3*size*ps + 2*i + 2*size   ] = (float) aux;
       aux = imag(in->fermion[ps][i].comp[2]) - (double) out[3*size*ps + 2*i + 2*size +1];
       out[offs + 3*size*ps + 2*i + 2*size +1] = (float) aux;
       }
    }
 #ifdef DEBUG_MODE
 cout << "\tterminated smartpack_multifermion"<<endl;
 #endif
 } 



void smartunpack_multifermion(MultiFermion *out, const float *in)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartunpak_multifermion ..."<<endl;
 #endif

 long int i;
 long int offs = 6*sizeh*GlobalParams::Instance().getNumPS();
 int ps;
 
 double d_re, d_im; 
 complex<REAL> aux[3]; 


 for(ps=0; ps<GlobalParams::Instance().getNumPS(); ps++)
    {
    for(i=0; i<sizeh; i++)
       {
       d_re =(double) in[3*size*ps + 2*i                      ];
       d_re+=(double) in[3*size*ps + 2*i               + offs ];
       d_im =(double) in[3*size*ps + 2*i           + 1        ];
       d_im+=(double) in[3*size*ps + 2*i           + 1 + offs ];
       aux[0]=complex<REAL>(d_re, d_im);

       d_re =(double) in[3*size*ps + 2*i + size               ];
       d_re+=(double) in[3*size*ps + 2*i + size        + offs ];
       d_im =(double) in[3*size*ps + 2*i + size    + 1        ];
       d_im+=(double) in[3*size*ps + 2*i + size    + 1 + offs ];
       aux[1]=complex<REAL>(d_re, d_im);

       d_re =(double) in[3*size*ps + 2*i + 2*size             ];
       d_re+=(double) in[3*size*ps + 2*i + 2*size      + offs ];
       d_im =(double) in[3*size*ps + 2*i + 2*size  + 1        ];
       d_im+=(double) in[3*size*ps + 2*i + 2*size  + 1 + offs ];
       aux[2]=complex<REAL>(d_re, d_im);

       out->fermion[ps][i]=Vec3(aux);
       }
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartunpak_multifermion"<<endl;
 #endif
 } 


// ShiftMultiFermion unpacker
// 6*sizeh*max_approx_order*no_ps*2 = 6*sizeh(fermion)*max_approx_order*no_ps*2(1double~2float)
void smartunpack_multishiftfermion(ShiftMultiFermion *out, const float *in,  int order)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartunpack_multishiftfermion ..."<<endl;
 #endif

 long int i, shift, ps;
 long int offs = 3*size*order*GlobalParams::Instance().getNumPS();
 double tempR, tempI;

 for(ps=0; ps<GlobalParams::Instance().getNumPS(); ps++)
    {
    for(shift=0; shift<order; shift++)
       {
       for(i=0; i<sizeh; i++)
          {
          tempR  = (double) in[       3*size*order*ps + 3*size*shift + 2*i            ];
          tempR += (double) in[offs + 3*size*order*ps + 3*size*shift + 2*i            ];
          tempI  = (double) in[       3*size*order*ps + 3*size*shift + 2*i          +1];
          tempI += (double) in[offs + 3*size*order*ps + 3*size*shift + 2*i          +1];
          out->fermion[ps][shift][i].comp[0] = complex<REAL>(tempR, tempI);

          tempR  = (double) in[       3*size*order*ps + 3*size*shift + 2*i +   size   ];
          tempR += (double) in[offs + 3*size*order*ps + 3*size*shift + 2*i +   size   ];
          tempI  = (double) in[       3*size*order*ps + 3*size*shift + 2*i +   size +1];
          tempI += (double) in[offs + 3*size*order*ps + 3*size*shift + 2*i +   size +1];
          out->fermion[ps][shift][i].comp[1] = complex<REAL>(tempR, tempI);
 
          tempR  = (double) in[       3*size*order*ps + 3*size*shift + 2*i + 2*size   ];
          tempR += (double) in[offs + 3*size*order*ps + 3*size*shift + 2*i + 2*size   ];
          tempI  = (double) in[       3*size*order*ps + 3*size*shift + 2*i + 2*size +1];
          tempI += (double) in[offs + 3*size*order*ps + 3*size*shift + 2*i + 2*size +1];
          out->fermion[ps][shift][i].comp[2] = complex<REAL>(tempR, tempI);
          }
       }
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartunpack_multishiftfermion"<<endl;
 #endif
 }


// create shift table
void make_shift_table(int table[8*size])
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside make_shift_table ..."<<endl;
 #endif

 long int i;

 for(i=0; i<size; i++)
    {
    table[i         ]= nnm[i][0];
    table[i +   size]= nnm[i][1];
    table[i + 2*size]= nnm[i][2];
    table[i + 3*size]= nnm[i][3];

    table[i + 4*size]= nnp[i][0];
    table[i + 5*size]= nnp[i][1];
    table[i + 6*size]= nnp[i][2];
    table[i + 7*size]= nnp[i][3];
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated make_shift_table"<<endl;
 #endif
 }


// pack traceless antihermitian matrix
void smartpack_tamatrix(float out[8*no_links], Ipdot *in)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartpack_tamatrix ..."<<endl;
 #endif

 long int i, mu;

 for(i=0; i<size;i++)
    {
    for(mu=0; mu<4; mu++)
       {
       out[4*i            +8*mu*size]=(float) real(in->ipdot[i+mu*size].comp[0][1]);
       out[4*i +1         +8*mu*size]=(float) imag(in->ipdot[i+mu*size].comp[0][1]);
       out[4*i +2         +8*mu*size]=(float) real(in->ipdot[i+mu*size].comp[0][2]);
       out[4*i +3         +8*mu*size]=(float) imag(in->ipdot[i+mu*size].comp[0][2]);
       out[4*i    +4*size +8*mu*size]=(float) real(in->ipdot[i+mu*size].comp[1][2]);
       out[4*i +1 +4*size +8*mu*size]=(float) imag(in->ipdot[i+mu*size].comp[1][2]);
       out[4*i +2 +4*size +8*mu*size]=(float) imag(in->ipdot[i+mu*size].comp[0][0]);
       out[4*i +3 +4*size +8*mu*size]=(float) imag(in->ipdot[i+mu*size].comp[1][1]);
       }
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartpack_tamatrix"<<endl;
 #endif
 }


// unpack traceless antihermitian matrix
void smartunpack_tamatrix(Ipdot *out, float in[8*no_links])
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartunpack_tamatrix ..."<<endl;
 #endif

 long int i, mu;
 complex<REAL> aux[3][3];

 for(i=0; i<size;i++)
    {
    for(mu=0; mu<4; mu++)
       {
       aux[0][1]=complex<REAL>(in[4*i                   + 8*mu*size], in[4*i + 1          + 8*mu*size]);
       aux[0][2]=complex<REAL>(in[4*i + 2               + 8*mu*size], in[4*i + 3          + 8*mu*size]);
       aux[1][2]=complex<REAL>(in[4*i          + 4*size + 8*mu*size], in[4*i + 1 + 4*size + 8*mu*size]);
       aux[0][0]=complex<REAL>(0.0, in[4*i + 2 + 4*size + 8*mu*size]);
       aux[1][1]=complex<REAL>(0.0, in[4*i + 3 + 4*size + 8*mu*size]);

       aux[2][2]=-aux[0][0]-aux[1][1];
       aux[1][0]=-conj(aux[0][1]);
       aux[2][0]=-conj(aux[0][2]);
       aux[2][1]=-conj(aux[1][2]);

       out->ipdot[i+mu*size]=Su3(aux);
       }
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartunpack_tamatrix"<<endl;
 #endif
 }


// pack traceless hermitian matrix
void smartpack_thmatrix(float out[8*no_links], Momenta *in)
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartpack_thmatrix ..."<<endl;
 #endif

 long int i, mu;

 for(i=0; i<size;i++)
    {
    for(mu=0; mu<4; mu++)
       {
       out[4*i            +8*mu*size]=(float) real(in->momenta[i+mu*size].comp[0][1]);
       out[4*i +1         +8*mu*size]=(float) imag(in->momenta[i+mu*size].comp[0][1]);
       out[4*i +2         +8*mu*size]=(float) real(in->momenta[i+mu*size].comp[0][2]);
       out[4*i +3         +8*mu*size]=(float) imag(in->momenta[i+mu*size].comp[0][2]);
       out[4*i    +4*size +8*mu*size]=(float) real(in->momenta[i+mu*size].comp[1][2]);
       out[4*i +1 +4*size +8*mu*size]=(float) imag(in->momenta[i+mu*size].comp[1][2]);
       out[4*i +2 +4*size +8*mu*size]=(float) real(in->momenta[i+mu*size].comp[0][0]);
       out[4*i +3 +4*size +8*mu*size]=(float) real(in->momenta[i+mu*size].comp[1][1]);
       }
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartpack_thmatrix"<<endl;
 #endif
 }


// unpack traceless hermitian matrix
void smartunpack_thmatrix(Momenta *out, float in[8*no_links])
 {
 #ifdef DEBUG_MODE
 cout << "DEBUG: inside smartunpack_thmatrix ..."<<endl;
 #endif

 long int i, mu;
 complex<REAL> aux[3][3];

 for(i=0; i<size;i++)
    {
    for(mu=0; mu<4; mu++)
       {
       aux[0][1]=complex<REAL>(in[4*i                   + 8*mu*size], in[4*i + 1          + 8*mu*size]);
       aux[0][2]=complex<REAL>(in[4*i + 2               + 8*mu*size], in[4*i + 3          + 8*mu*size]);
       aux[1][2]=complex<REAL>(in[4*i          + 4*size + 8*mu*size], in[4*i + 1 + 4*size + 8*mu*size]);
       aux[0][0]=complex<REAL>(in[4*i + 2 + 4*size + 8*mu*size], 0.0);
       aux[1][1]=complex<REAL>(in[4*i + 3 + 4*size + 8*mu*size], 0.0);

       aux[2][2]=-aux[0][0]-aux[1][1];
       aux[1][0]=conj(aux[0][1]);
       aux[2][0]=conj(aux[0][2]);
       aux[2][1]=conj(aux[1][2]);

       out->momenta[i+mu*size]=Su3(aux);
       }
    }

 #ifdef DEBUG_MODE
 cout << "\tterminated smartunpack_thmatrix"<<endl;
 #endif
 }

