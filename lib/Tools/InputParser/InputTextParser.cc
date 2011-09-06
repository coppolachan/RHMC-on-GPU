/*
InputTextParser.cpp

Parser for plain text input file

2011  Guido Cossu: cossu@post.kek.jp

*/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <stdlib.h>

#include "InputTextParser.h"
#include "StringTools.h"




std::string removeSpaces( std::string stringIn )
{
  std::string::size_type pos = 0;
  bool spacesLeft = true;
  
  while( spacesLeft )
    {
      pos = stringIn.find(" ");
      if( pos != std::string::npos )
	stringIn.erase( pos, 1 );
      else
	spacesLeft = false;
    }
  
  return stringIn;
} 

std::string Parser(std::istream& inFile) {
  std::string nextLine;
  getline(inFile, nextLine, '\n');

  while ( inFile && (nextLine.length() == 0 ) ) { //skip empty lines
    getline(inFile, nextLine, '\n');
  }

  return nextLine;
}



// Default constructor
InputTextParser::InputTextParser(std::string inputFileName) {
  FileName = inputFileName;
}


int InputTextParser::ParseFile() {
  std::ifstream File(FileName.c_str());
  
  if ( File.fail() ){
    std::cerr << "Error - file not found : " << File << " . Check file name. " << std::endl;
  }

  std::string Next = Parser(File);

  while ( File ) {
    ProcessParameter(Next);
    Next = Parser( File );
  }

  File.close();

}


void InputTextParser::ProcessParameter(std::string ParamLine) {
  std::string ParamString;
  std::string ParamValue;  
  std::istringstream In(ParamLine);

  getline(In, ParamString, ' '); //space delimiter
  getline(In, ParamValue, '\n');

  ParamString = removeSpaces(ParamString);
  ParamValue = removeSpaces(ParamValue);

  if (ParamString.compare(1,1,"#") > 0) {
    ParameterMap.insert(std::pair<std::string, std::string>(ParamString,ParamValue));

#ifdef DEBUG
    std::cout << "Map size: " << (int) ParameterMap.size() << " elements. ";
    std::cout << "Key: " << ParamString << " Value: "<< ParameterMap[ParamString] << std::endl;
#endif 
  }
}

bool InputTextParser::CheckKey(std::string Key) {
  //Test the existence of Key
  std::map< std::string, std::string >::iterator it;
  it = ParameterMap.find(Key);

  if (it == ParameterMap.end() ) {
    std::cerr << Key <<": No entry with that name, check your input file" << std::endl;
    return 1;
  }

  return 0;
}

void InputTextParser::List() {
  std::map<std::string,std::string>::iterator it;


  std::cout << "List of Parameters in input file: "<< 
    FileName  << std::endl;
  std::cout << "Number of parameters found : " <<
    (int) ParameterMap.size() << std::endl << std::endl;

  for ( it=ParameterMap.begin() ; it != ParameterMap.end(); it++ )
    std::cout << (*it).first << "   =>   " << (*it).second << std::endl;
}

void InputTextParser::get(const std::string ParameterName, double&val) {
  std::string StringVal;
  if (!CheckKey(ParameterName)) {
    StringVal = ParameterMap[ParameterName];
    
    from_string(val, StringVal, std::dec);
  }
  else 
    exit(1);
  
}

void InputTextParser::get(const std::string ParameterName, int&val) {
  std::string StringVal;
  if (!CheckKey(ParameterName)) {
    StringVal = ParameterMap[ParameterName];
    
    from_string(val, StringVal, std::dec);
  }
  else 
    exit(1);
  
}

void InputTextParser::get(const std::string ParameterName, bool&val) {
  std::string StringVal;
  if (!CheckKey(ParameterName)) {
    StringVal = ParameterMap[ParameterName];
    
    val = to_bool(StringVal);
  }
  else 
    exit(1);
  
}

void InputTextParser::get(const std::string ParameterName, std::string&val) {
  
  if (!CheckKey(ParameterName)) {
    val = ParameterMap[ParameterName];
  }
  else 
    exit(1);
  
}
