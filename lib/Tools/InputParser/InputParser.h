#ifndef INPUT_PARSER_H_
#define INPUT_PARSER_H_

/*
InputParser.h

Parser for generic input file
Abstract base class

2011  Guido Cossu: cossu@post.kek.jp

*/

#include <string>


class InputParser{
protected:
  std::string FileName;

public:
  virtual int ParseFile() {};
  virtual void List(){};
  //virtual functions cannot be templated 
  virtual void get(const std::string ParameterName, double &val){};
  virtual void get(const std::string ParameterName, int &val){};
  virtual void get(const std::string ParameterName, bool &val){};
  virtual void get(const std::string ParameterName, std::string &val){};
};



#endif
