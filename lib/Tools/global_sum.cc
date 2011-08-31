// Routine to sum the entries of vec[i]
// after the routine has finished sum=vec[0]
template<typename T> void global_sum(T *vec, long int i)
  {
  if(i==1)
    {
    // base case
    }
  else
    {
    if(i%2==0)
      {
      long int j, k;

      j=i/2;
      for(k=0; k<j; k++)
         {
         vec[k]+=vec[k+j];
         }
      global_sum(vec, j);
      }
    else
      {
      long int j1, j2, k;
  
      j1=i/2;
      j2=j1+1;
      for(k=0; k<j1; k++)
         {
         vec[k]+=vec[k+j2];
         }
      global_sum(vec, j2);
      }
    }
  }
