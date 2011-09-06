#ifndef STRING_TOOLS_H_
#define STRING_TOOLS_H_

/*
Some useful tools for manipulating strings

*/

#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>


template <class T>
bool from_string(T& t, 
                 const std::string& s, 
                 std::ios_base& (*f)(std::ios_base&))
{
  std::istringstream iss(s);
  return !(iss >> f >> t).fail();
};

bool to_bool(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    std::istringstream is(str);
    bool b;
    is >> std::boolalpha >> b;
    return b;
}



#endif
