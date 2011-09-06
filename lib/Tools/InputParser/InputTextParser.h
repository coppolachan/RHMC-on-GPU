#ifndef INPUT_TEXT_PARSER_H_
#define INPUT_TEXT_PARSER_H_

/*
InputTextParser.h

Parser for plain text input file

2011  Guido Cossu: cossu@post.kek.jp

*/


#include <fstream>
#include <string>
#include <map>
#include "InputParser.h"



class InputTextParser: public InputParser{

  std::map<std::string,std::string> ParameterMap;

  int ParseFile();
  void ProcessParameter(std::string ParamLine);
  bool CheckKey(std::string Key);

public:
  InputTextParser(std::string TextInputFileName);
  void List();
  void get(const std::string ParameterName, double&val);
  void get(const std::string ParameterName, int &val);
  void get(const std::string ParameterName, bool &val);
  void get(const std::string ParameterName, std::string &val);
};



#endif
