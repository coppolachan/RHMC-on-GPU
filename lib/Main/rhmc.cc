// rhmc.cc: main source code

/*!
  Source code of the main hmc program

 */
#include <string>

#define DEFINE_GLOBALS
#include "include/rhmc_include.h"
#include "include/global_var.h"
#include "include/init.h"
#include "include/parameters.h"
#include "include/tools.h"
#include "include/update.h"
#include "include/measurement.h"


#include "include/gpu.h"



using namespace std;

void simulation_details(void);

int main(void)
   {
     
     int init_error, update_iteration_old, loc_update;   
     int acc_test, acc_run;
     int stop_cond=1;
     clock_t time_start, time_finish;
     ofstream out_file;
     
     init_error = initialize();
     
     if(init_error==0)
       {
	 // accepted trajectories in this run
	 acc_run=0; 

	 // update_iterations_old = # of updates in previous simulations
	 update_iteration_old=update_iteration;  

	 loc_update=0;
	 
	 simulation_details();
	 
	 
	 // STANDARD_UPDATE
#ifdef STANDARD_UPDATE
	 if(GlobalParams::Instance().getStartState()==2) 
	   {
	     out_file.open(QUOTEME(DATA_FILE), ios::app); 
	   }
	 else
	   {
	     out_file.open(QUOTEME(DATA_FILE), ios::out); 
	     out_file << "## "<< nx << " " << ny << " " << nz << " "<< nt << " ";
	     out_file << GlobalParams::Instance().getNf() << " " << 
	       GlobalParams::Instance().getMass() << " " << GlobalParams::Instance().getBeta() << "\n";
	   }
	 out_file.precision(12);
	 
	 cout << "\n=== STANDARD UPDATE ===\n";
	 while(update_iteration-update_iteration_old<GlobalParams::Instance().getRunUpdateIterations() && stop_cond==1)
	   {
	     cout << "\n--- Iteration number " << update_iteration << "  [ still ";
	     cout << GlobalParams::Instance().getRunUpdateIterations() - update_iteration + update_iteration_old << 
	       " iterations to end ]---"<<endl;
	     
#ifdef USE_GPU
	     gpu_init0();
#endif
	     
#ifndef NO_MEASURE
	     if(update_iteration % GlobalParams::Instance().getMeasInterval()==0)
	       {
		 time_start=clock();
		 meas(out_file); 
		 time_finish=clock();
		 cout << "time for measurements = " << 
		   ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
	       }
#endif
	     
	     time_start=clock();
	     update(acc_test);
	     time_finish=clock();
	     
	     if(acc_test==1) 
	       { 
		 acc_run+=acc_test;
		 loc_update++;
		 cout << "Accepted\t\t["<< acc_run <<"/"<< loc_update <<"]\n";
	       }
	     if(acc_test==0)
	       {
		 acc_run+=acc_test;
		 loc_update++;
		 cout << "Rejected\t\t["<< acc_run <<"/"<< loc_update <<"]\n";
	       }
	     if(acc_test==-1)
	       {
		 cout << "Thermalization update\n";
	       }
	     cout << "time for update = " << 
	       ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
	     
	     update_iteration++;
	     
	     // write configuration to file
	     if(update_iteration % GlobalParams::Instance().getSaveInterval() ==0)  
	       {
		 time_start=clock();
		 if(update_iteration-update_iteration_old!=GlobalParams::Instance().getRunUpdateIterations())
		   {
		     gauge_conf->write();
		   }
		 else
		   {
		     gauge_conf->write_last();
		   }
		 time_finish=clock();
		 cout << "Configuration saved (in "<< 
		   ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.)\n";
	       }
	     
	     // if exist the file "stop" end update
	     stop_cond=look_for_file("stop");
	     
	     // write configuration to file
	     if(stop_cond==0)  
	       {
		 time_start=clock();
		 gauge_conf->write_last();
		 time_finish=clock();
		 cout << "Configuration saved (in "<< 
		   ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.)\n";
	       }
	     
	     // write configuration to file
	     if(update_iteration %GlobalParams::Instance().getSaveInterval() != 0 && 
		update_iteration-update_iteration_old==GlobalParams::Instance().getRunUpdateIterations())
	       {
		 time_start=clock();
		 gauge_conf->write_last();
		 time_finish=clock();
		 cout << "Configuration saved (in "<< 
		   ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.)\n";
	       }
	   }
	 
	 out_file.close();
#endif
	 
	 
	 // PARAMETER_TEST UPDATE
#ifdef PARAMETER_TEST
	 cout << "\n=== PARAMETER TEST UPDATE ===\n";
	 while(update_iteration-update_iteration_old<GlobalParams::Instance().getRunUpdateIterations() && stop_cond==1)
	   {
	     cout << "\n--- Iteration number " << update_iteration << "  [ still ";
	     cout << GlobalParams::Instance().getRunUpdateIterations() - update_iteration + update_iteration_old << 
	       " iterations to end ]---"<< endl;
	     
#ifdef USE_GPU
	     gpu_init0();
#endif
	     
	     update(acc_test);
	     
	     if(acc_test==1) 
	       { 
		 acc_run+=acc_test;
		 loc_update++;
		 cout << "Accepted\t\t["<< acc_run <<"/"<< loc_update <<"]\n";
	       }
	     if(acc_test==0)
	       {
		 acc_run+=acc_test;
		 loc_update++;
		 cout << "Rejected\t\t["<< acc_run <<"/"<< loc_update <<"]\n";
	       }
	     if(acc_test==-1)
	       {
		 cout << "Thermalization update\n";
	       }
	     
	     // if exist the file "stop" end update
	     stop_cond=look_for_file("stop");
	     
	     update_iteration++;
	   }
#endif
	 
	 
	 // REVERSIBILITY_TEST UPDATE
#ifdef REVERSIBILITY_TEST
	 cout << "\n=== REVERSIBILITY_TEST UPDATE ===\n";
	 while(update_iteration-update_iteration_old<GlobalParams::Instance().getRunUpdateIterations() && stop_cond==1)
	   {
	     cout << "\n--- Iteration number " << update_iteration << "  [ still ";
	     cout << GlobalParams::Instance().getRunUpdateIterations() - update_iteration + update_iteration_old << 
	       " iterations to end ]---"<<endl;
	     
#ifdef USE_GPU
	     gpu_init0();
#endif
	     
	     time_start=clock();
	     update(acc_test);
	     time_finish=clock();
	     cout << "time for update = " << 
	       ((REAL)(time_finish)-(REAL)(time_start))/CLOCKS_PER_SEC << " sec.\n";
	     if(acc_test==1) 
	       { 
		 acc_run+=acc_test;
		 loc_update++;
		 cout << "Accepted\t\t["<< acc_run <<"/"<< loc_update <<"]\n";
	       }
	     if(acc_test==0)
	       {
		 acc_run+=acc_test;
		 loc_update++;
		 cout << "Rejected\t\t["<< acc_run <<"/"<< loc_update <<"]\n";
	       }
	     if(acc_test==-1)
	       {
		 cout << "Thermalization update\n";
	       }
	     
#ifdef USE_GPU
	     gpu_init0();
#endif
	     
	     rev_update();
	     
	     // if exist the file "stop" end update
	     stop_cond=look_for_file("stop");
	     
	     update_iteration++;
	   }
#endif
       }
     
     finalize();
   }



////////////// END OF MAIN


void simulation_details(void)
{
  cout <<"\n=== SIMULATION PARAMETERS ==\n\n";
  
  cout <<"Simulation using "<< QUOTEME(REAL) <<"\n\n";
  
  cout <<"Number of threads for block "<< QUOTEME(NUM_THREADS) <<"\n\n";
  
  cout <<"Number of thermalization updates "<< GlobalParams::Instance().getThermUpdates() <<"\n\n";
  
  cout <<"Lattice dimensions: nx = "<<nx<<"  ny = "<<ny<<"  nz = "<<nz<<"  nt = "<<nt<<"\n";
  
#ifndef PURE_GAUGE
  cout <<"Number of flavours = "<< GlobalParams::Instance().getNf()<<"\n";
  cout <<"Quark mass =  "<< GlobalParams::Instance().getMass()<<"\n";
#else
  cout <<"Pure gauge simulation\n";
#endif
  
  cout <<"beta = "<<GlobalParams::Instance().getBeta()<<"\n";
  
#ifdef IM_CHEM_POT
  cout <<"immaginary chemical potential = "<<immu<<"\n";
#endif
  
  if(GlobalParams::Instance().getStartState()==0) cout << "Cold start\n";
  if(GlobalParams::Instance().getStartState()==1) cout << "Hot start\n";
  if(GlobalParams::Instance().getStartState()==2) cout << "Previous simulation continuation\n";
  cout <<"\n";
  
  if( GlobalParams::Instance().getMultistep().compare("NO_MULTISTEP")==0)
    {
      cout <<"Multistep integrator NOT used\n";
    }
  else
    {
      if(GlobalParams::Instance().getMultistep().compare("2MN_MULTISTEP")==0)
	{
	  cout <<"Using 2MN multistep integrator\n";
	}
      if(GlobalParams::Instance().getMultistep().compare("4MN_MULTISTEP")==0)
	{
	  cout <<"Using 4MN multistep integrator\n";
	}
      cout <<"Gauge scale for multistep integrator = "<<
	GlobalParams::Instance().getGaugeTimeScale()<<"\n";
    }
  
  cout <<"Number of MD steps for trajectory (time length = 1.0) = "<<
    GlobalParams::Instance().getNumMD()<<"\n\n";
  
#ifndef PURE_GAUGE
  cout <<"Number of pseudofermions = "<< no_ps<<"\n\n";
#endif
  
  cout <<"Residue for CG inverter in metropolis test = "<< 
    GlobalParams::Instance().getResidueMetro() <<"\n";
  cout <<"Residue for CG inverter in molecular dynamics = "<< 
    GlobalParams::Instance().getResidueMD() <<"\n";
#ifdef USE_GPU
  cout <<"Double precision inverter used if residue < " << 
    GlobalParams::Instance().getInvSingleDoublePrec() <<"\n";
#endif
  cout <<"\n";
  
  cout <<"Number of random vectors for chiral measurements = "<<
    GlobalParams::Instance().getRandVect() <<"\n";
  
#ifdef USE_GPU
  cout <<"\n";
  GPUDeviceInfo(GlobalParams::Instance().getGPUDevice());
#endif
  
  cout <<endl;
}
