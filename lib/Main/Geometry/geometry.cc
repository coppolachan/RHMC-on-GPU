#include <iostream>
#include "include/global_var.h"

// superindex
long int snum(int x, int y, int z, int t)
  {
  long int ris;

  ris=x+y*vol1+z*vol2+t*vol3;
  return ris/2;   // <---  /2 Pay attention to even/odd  (see init_geo) 
  }





//   links used according to this scheme
//
//   0            size         size2         size3         no_links
//   |------|------|------|------|------|------|------|------|
//      e      o      e       o     e      o      e      o
//        x-dir         y-dir         z-dir         t-dir

// initialize geometry
// periodic spatial bc are always assumed
void init_geo(void)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside init_geo ..."<<endl;
  #endif

  int even;
  int x, y, z, t, xm, ym, zm, tm, xp, yp, zp, tp;
  long int num;
  int sum, rest;
  
  // allocate nnp[size][4]
  nnp = new long int * [size];
  for(num=0; num<size; num++)
     {
     nnp[num] = new long int [4];
     }

  // allocate nnm[size][4]
  nnm = new long int * [size];
  for(num=0; num<size; num++)
     {
     nnm[num] = new long int [4];
     }

  for(t=0; t<nt; t++)
     {
     tp=t+1;
     tm=t-1;
     if(t==nt-1) tp=0;
     if(t==0) tm=nt-1;

     for(z=0; z<nz; z++)
        {
        zp=z+1;
        zm=z-1;
        if(z==nz-1) zp=0;
        if(z==0) zm=nz-1;

        for(y=0; y<ny; y++)
           {
           yp=y+1;
           ym=y-1;
           if(y==ny-1) yp=0;
           if(y==0) ym=ny-1;

           for(x=0; x<nx; x++)
              {
              xp=x+1;
              xm=x-1;
              if(x==nx-1) xp=0;
              if(x==0) xm=nx-1;

              // the even sites get the lower half (     0 --> sizeh-1 )
              // the odd  sites get the upper half ( sizeh --> size -1 )
              
              sum = x+y+z+t;
              even = sum%2;        // even=0 for even sites
                                   // even=1 for odd sites

              num = even*sizeh + snum(x,y,z,t);

              // NEXT-NEIGHBOURS DEFINITION

              // x-dir             
              nnp[num][0]=(1-even)*sizeh + snum(xp,y,z,t);
              nnm[num][0]=(1-even)*sizeh + snum(xm,y,z,t);
             
              //y-dir
              nnp[num][1]=(1-even)*sizeh + snum(x,yp,z,t);
              nnm[num][1]=(1-even)*sizeh + snum(x,ym,z,t);

              //z-dir
              nnp[num][2]=(1-even)*sizeh + snum(x,y,zp,t);
              nnm[num][2]=(1-even)*sizeh + snum(x,y,zm,t);
 
              //t-dir
              nnp[num][3]=(1-even)*sizeh + snum(x,y,z,tp);
              nnm[num][3]=(1-even)*sizeh + snum(x,y,z,tm);


              // ETA definitions
              
              // x-dir
              eta[num]=1;

              // y-dir
              sum=x;
              rest=sum%2;
              if(rest==0)  
                {
                eta[num+size]=1;
                }
              else
                {
                eta[num+size]=-1;
                }

              // z-dir
              sum=x+y;
              rest=sum%2;
              if(rest==0)  
                {
                eta[num+size2]=1;
                }
              else
                {
                eta[num+size2]=-1;
                }

              // t-dir
              sum=x+y+z;
              rest=sum%2;
              if(rest==0)  
                {
                eta[num+size3]=1;
                }
              else
                {
                eta[num+size3]=-1;
                }
              if(GlobalParams::Instance().getFermTempBC() == 0)
                {
                if(t==nt-1)        // antiperiodic temporal b.c. for fermions
                  {
                  eta[num+size3]=-eta[num+size3];
                  }
                }
              }
           } 
        }
     }
  #ifdef DEBUG_MODE
  cout << "\tterminated init_geo"<<endl;
  #endif
  }



// clear geometry
void end_geo(void)
  {
  #ifdef DEBUG_MODE
  cout << "DEBUG: inside end_geo ..."<<endl;
  #endif

  long int num;

  for(num=0; num<size; num++)
     {
     delete [] nnm[num];
     }
  delete [] nnm;

  for(num=0; num<size; num++)
     {
     delete [] nnp[num];
     }
  delete [] nnp;

  #ifdef DEBUG_MODE
  cout << "\tterminated end_geo"<<endl;
  #endif
  }
