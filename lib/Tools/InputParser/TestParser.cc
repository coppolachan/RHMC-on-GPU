// Tester for the InputParser class

#include "InputParser.h"
#include "InputTextParser.h"
#include <iostream>

int main() {
  InputParser *Input = new InputTextParser("input_parameters");

  Input->ParseFile();

  Input->List();

  double beta;
  int par;
  bool parbool;
  std::string valstring;
  Input->get("Beta",beta);

  std::cout << beta << std::endl;

}
