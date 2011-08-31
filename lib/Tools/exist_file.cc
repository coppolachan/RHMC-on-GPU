// return 0 if file exists else 1

#include <string>
#include<sys/stat.h>

using namespace std;

int look_for_file(string str_filename) 
    {
    struct stat file_info;
    int a;

    // Attempt to get the file attributes
    a = stat(str_filename.c_str(),&file_info);
    if(a == 0) 
      {
      // We were able to get the file attributes
      // so the file obviously exists.
      return 0;
      }  
    else 
      {
      // We were not able to get the file attributes.
      // This may mean that we don't have permission to
      // access the folder which contains this file. If you
      // need to do that level of checking, lookup the
      // return values of stat which will give you
      // more details on why stat failed.
      return 1;
      }
    }
