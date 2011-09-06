#include <iostream>
#include <string>
#include "include/parameters.h"
#include <cstdlib>
#include "lib/Tools/InputParser/InputParser.h"



void Params::setParams(InputParser &Input) {
  Input.ParseFile();

  Input.get("NumFlavors",no_flavors);

  Input.get("Beta",beta);
  beta_by_three = beta/3.0;

  Input.get("Mass",mass);
  one_by_mass=1.0/mass;
  mass2=mass*mass;

  Input.get("Rand_Seed",rand_seed);

  Input.get("FermTempBC", ferm_temp_bc);

  Input.get("StartConf", starting_state);

  Input.get("RunUpdateIterations",run_update_iterations);
  Input.get("ThermUpdates",therm_updates);
  Input.get("SaveConfInterval", save_interval);
  Input.get("MeasureInterval", meas_interval);

  Input.get("NumMD", no_md);
  Input.get("UseMultiStep",use_multistep);
  Input.get("GaugeTimeScale",gauge_scale);
  //Check allowed multistep modes
  if (use_multistep.compare("NO_MULTISTEP") &&
      use_multistep.compare("2MN_MULTISTEP") &&
      use_multistep.compare("4MN_MULTISTEP")) {
    std::cerr<< "Undefined multistep choice : " << use_multistep << std::endl;
    std::cerr<< "Check input file "<< std::endl;
    exit(1);
  }

  epsilon=1.0/((REAL) no_md);
  ieps=std::complex<REAL>(0.0, epsilon);
  iepsh=ieps*std::complex<REAL>(0.5,0.0);

  Input.get("RandVect", rand_vect);

  Input.get("MaxCG",max_cg);
  Input.get("NumPseudoFerm",no_ps);
  Input.get("GMP_RemezPrec",gmp_remez_precision);

  Input.get("InvSingleDblPrec",inv_single_double_prec);

  Input.get("ApproxMetro",approx_metro);
  Input.get("LambdaMinMetro",lambda_min_metro);
  Input.get("ResidueMetro",residue_metro);

  Input.get("ApproxMD",approx_md);
  Input.get("LambdaMinMD",lambda_min_md);
  Input.get("ResidueMD",residue_md);

  gpu_device_to_use = 0; //default choice
  Input.get("GPUDevice",gpu_device_to_use);
}

void Params::listParams() {
  //List all parameters on screen

  std::cout << "NumFlavors    : "<< no_flavors << std::endl;
  std::cout << "Beta          : "<< beta << std::endl;
  std::cout << "Mass          : "<< mass << std::endl;
  std::cout << "Random Seed   : "<< rand_seed << std::endl;
  if(ferm_temp_bc) {
    std::cout << "Fermionic Temporal BC   : PERIODIC" << std::endl;
  }
  else {
    std::cout << "Fermionic Temporal BC   : ANTI-PERIODIC" << std::endl;
  }
  std::cout << "Starting State   : "<< starting_state << std::endl;
  std::cout << "Run updates                 : "<< run_update_iterations << std::endl;
  std::cout << "Thermalization updates      : "<< therm_updates << std::endl;
  std::cout << "Save interval (#conf)       : "<< save_interval << std::endl;
  std::cout << "Measure interval (#conf)    : "<< meas_interval << std::endl;

  std::cout << "Number of MD steps          : "<< no_md << std::endl;
  std::cout << "UseMultistep                : "<< use_multistep << std::endl;
  std::cout << "Gauge Time Scale            : "<< gauge_scale << std::endl;  
  std::cout << "Number of random vectors for measurements    : "<< rand_vect << std::endl; 

  std::cout << "Max CG iterations           : "<< max_cg << std::endl;
  //std::cout << "Number of pseudofermions    : "<< no_ps << std::endl;
  std::cout << "GMP Remez precision goal    : "<< gmp_remez_precision << std::endl;
  std::cout << "InverseSingleDblePrec       : "<< inv_single_double_prec << std::endl; 

  //  std::cout << "Number of polynomials(Metro): "<< approx_metro << std::endl;
  std::cout << "Lambda minimum (Metropolis) : "<< lambda_min_metro << std::endl;
  std::cout << "Residue (Metropolis)        : "<< residue_metro << std::endl;

  //  std::cout << "Number of polynomials(MD): "<< approx_md << std::endl;
  std::cout << "Lambda minimum (MolDyn) : "<< lambda_min_md << std::endl;
  std::cout << "Residue (MolDyn)        : "<< residue_md << std::endl;

  std::cout << ":::::: GPU Device       : "<< gpu_device_to_use << std::endl;
}


void ChemPotParams::setParams(InputParser &Input){
  UseState = false; //default do not use chemical potential
  Input.get("ChemPotential",use_chem_potential);
  if (use_chem_potential.compare("NO_CHEMPOT") &&
      use_chem_potential.compare("USE_CHEMPOT")) {
    std::cerr<< "Undefined chemical potential choice : " << use_chem_potential 
	     << std::endl;
    std::cerr<< "Check input file "<< std::endl;
    exit(1);
  }

  //if (use_chem_potential.compare("USE_CHEMPOT"))
  //temporary workaround to use just the global define in global_macro.h
#ifdef IM_CHEM_POT
  UseState = true;
#endif
  //end of workaround.

  Input.get("ImMu",immu);

  eim_cos=cos(immu);
  eim_sin=sin(immu);

}
void ChemPotParams::listParams() {
  //Commented because it is now yet available to use
  //  std::cout << "Use chemical potential       : "<< use_chem_potential << std::endl;  
  std::cout << "Imaginary chemical potential : "<< immu << std::endl;
}
