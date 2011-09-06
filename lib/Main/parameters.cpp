#include <iostream>
#include "include/parameters.h"

#include "lib/Tools/InputParser/InputParser.h"

int Params::no_flavors;

void Params::setParams(InputParser &Input) {
  Input.ParseFile();
  Input.get("NumFlavors",no_flavors);
}

void Params::listParams() {

  std::cout << "NumFlavors : "<< no_flavors << std::endl;


}
