#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TVector3.h"

using namespace std;

enum Particle {
  unclassified = 0,
  proton = 1,
  kaon = 2,
  pion = 3,
  other = 4
};
int nCubletsX = 10, nCubletsY = 10, nCubletsZ = 10;
int nCellsXY = 10;
int nCellsZ  = 10;
double cellSizeXY = 3; //mm
double cellSizeZ  = 12; //mm
double deltaE_vtx_thr = -50e3; // MeV, threshold of energy loss to be considered
                               // primary vertex of the event

// input tree variables
int i_evt;
int n_int;
vector<int>*    pdg;
vector<double>* edep;
vector<double>* deltae;
vector<double>* glob_t;
vector<int>*    cublet_idx;
vector<int>*    cell_idx;


void genPhotonTree(string filename, string treename, string outputFilePath,
                   vector<vector<vector<double>>> angle_matrix,
                   vector<vector<vector<double>>> time_matrix,
                   int verbose=0, bool primary_only=true, int max_event=1000) {

  auto start_time = std::chrono::high_resolution_clock::now();

  size_t name_start = filename.find_last_of('/');
  size_t name_end   = filename.find_last_of('.');
  string name = filename.substr(name_start+1, name_end-name_start-1);

  // set useful constant values
  int n_sensors = nCellsXY*nCellsZ;
  int nCublets = nCubletsX*nCubletsY*nCubletsZ;
  double lightyield = 200; // ph/MeV 
  double max_t = 20; // ns
  double dt = 0.2; // ns
  int timesteps = max_t/dt;


  // get simulation results tree from file
  TFile* file = TFile::Open(filename.c_str()); 
  TTree* tree = (TTree*) file->Get(treename.c_str())->Clone(); 

  tree->SetBranchAddress("Tevent_id",       &i_evt);
  tree->SetBranchAddress("Tinteractions_in_event", &n_int);
  tree->SetBranchAddress("Tpdg",            &pdg);
  tree->SetBranchAddress("Tedep",           &edep);
  tree->SetBranchAddress("Tdeltae",         &deltae);
  tree->SetBranchAddress("Tglob_t",         &glob_t);
  tree->SetBranchAddress("Tcublet_idx",     &cublet_idx);
  tree->SetBranchAddress("Tcell_idx",       &cell_idx);

  
  vector<vector<vector<int>>> photon_matrix(nCublets,  vector<vector<int>>(
                                            timesteps, vector<int>(
                                            n_sensors, 0)));
  vector<double> dEmax(nCublets, 0.0);                   // maximum energy diff between step beginning and end...
  vector<double> Etot(nCublets, 0.0);                    // total energy released...
  vector<TVector3> Ecentroid(nCublets, TVector3(0,0,0)); // (weighted) centroid of energy depositions...
  vector<double> sigmaR(nCublets, 0.0);                  // (weighted) energy dispersion on xy plane...
  vector<double> sigmaZ(nCublets, 0.0);                  // (weighted) energy dispersion along z...
  vector<Particle> p(nCublets, unclassified);            // particle classification...
  vector<int> Nint(nCublets, 0);                         // number of interactions...
  vector<int> pdg_max(nCublets, 0);                      // pdg encoding of primary particle...
                                                         // per cublet 

  ofstream outfile;
  if(primary_only){
    outfile.open(outputFilePath + name + ".dat", std::ios::binary);
    if (!outfile) {
      cerr << "Error opening file for writing!" << endl;
      return;
    }
  }

  // loop over events
  int n_evts = (int) tree->GetEntries(); // should be 1000
  if(max_event < n_evts){
    n_evts = max_event;
  }
  for (int i = 0; i < n_evts; i++) {

    if(verbose) {
      cout << "\tProcessing event " << i << "..." << endl;
    }

    tree->GetEntry(i);

    // reset values
    for(auto& matrix: photon_matrix) {
      for(auto& timestep: matrix) {
        fill(timestep.begin(), timestep.end(), 0.0);
      }
    }
    fill(dEmax.begin(),     dEmax.end(),     0.0);
    fill(Etot.begin(),      Etot.end(),      0.0);
    fill(Ecentroid.begin(), Ecentroid.end(), TVector3(0,0,0));
    fill(sigmaR.begin(),    sigmaR.end(),    0.0);
    fill(sigmaZ.begin(),    sigmaZ.end(),    0.0);
    fill(p.begin(),         p.end(),         unclassified);
    fill(Nint.begin(),      Nint.end(),      0);
    fill(pdg_max.begin(),   pdg_max.end(),   0);

    // primary vertex identification variables
    double dE_primary = 0;
    int primary_peak_cub = -1;
    
    // loop over interaction per event
    for (int j = 0; j < n_int; j++) {

      // check if energy has been deposited before max time
      double E  = (*edep)[j];
      double t0 = (*glob_t)[j];
      double dE = (*deltae)[j];
      if (E > 0 && t0 < max_t) {

        int cub_i = (*cublet_idx)[j];

        // update total energy
        Etot[cub_i] += E;

        // update number of interactions
        Nint[cub_i] += 1;

        // check if vertex is primary
        if(dE < deltaE_vtx_thr && dE < dE_primary) { // less than, as they are negative
          primary_peak_cub = cub_i;
          if(verbose > 1){
            cout << "\tPrimary vtx - pdg: " << (*pdg)[j]  << " - E: " << E << " - dE: " << dE << ";";
          }
        }

        // check maximum energy difference inside cublet
        if(dE < dEmax[cub_i]) {
          // update value
          dEmax[cub_i] = dE; 

          // store info on particle
          int part_ID = (*pdg)[j];
          pdg_max[cub_i] = part_ID;
          switch(part_ID) {
            case 2212:
              p[cub_i] = proton;
              break;
            case 321:
              p[cub_i] = kaon;
              break;
            case -321:
              p[cub_i] = kaon;
              break; 
            case 211:
              p[cub_i] = pion;
              break;
            case -211:
              p[cub_i] = pion;
              break; 
            default:
            p[cub_i] = other;
          }
        }

        int cell_i = (*cell_idx)[j];
        int z_idx =  cell_i/(nCellsXY*nCellsXY);           // i/(x*y)
        int y_idx = (cell_i%(nCellsXY*nCellsXY))/nCellsXY; // (i%(x*y))/x
        int x_idx =  cell_i%nCellsXY;                      // i%x

        // update centroid
        Ecentroid[cub_i].SetXYZ(Ecentroid[cub_i].X() + x_idx*E,
                                Ecentroid[cub_i].Y() + y_idx*E,
                                Ecentroid[cub_i].Z() + z_idx*E);

        
        // compute photon arriving to sensors
        double ph_emitted = E*lightyield;

        // loop over sensors
        for (int i_sz = 0; i_sz < nCellsZ; i_sz++) {
          for (int i_sx = 0; i_sx < nCellsXY; i_sx++) {
            int sensor_i = i_sz*nCellsXY+i_sx;
            int n_photon = round(ph_emitted*angle_matrix[abs(i_sx - x_idx)][y_idx][abs(i_sz - z_idx)]);
            double time = (t0+time_matrix[abs(i_sx - x_idx)][y_idx][abs(i_sz - z_idx)]);
            int step = time/dt;
            if(time < max_t) {
              photon_matrix[cub_i][step][sensor_i] += n_photon;
            }
          }
        }
      }
    }

    // correct centroid estimation
    for(int i_cub = 0; i_cub < nCublets; i_cub++) {
      if(!(Etot[i_cub] > 0)){
        Ecentroid[i_cub].SetXYZ(Ecentroid[i_cub].X()/Etot[i_cub],
                                Ecentroid[i_cub].Y()/Etot[i_cub],
                                Ecentroid[i_cub].Z()/Etot[i_cub]);
      }
    }

    // compute energy dispersions
    for (int j = 0; j < n_int; j++) {
      int cub_i = (*cublet_idx)[j];

      // check if energy has been released in the cublet, otherwise skip
      if((primary_only && (cub_i != primary_peak_cub)) || !(Etot[cub_i] > 0)) continue;
      
      double E  = (*edep)[j];
      int cell_i = (*cell_idx)[j];
      int z_idx =  cell_i/(nCellsXY*nCellsXY);           // i/(x*y)
      int y_idx = (cell_i%(nCellsXY*nCellsXY))/nCellsXY; // (i%(x*y))/x
      int x_idx =  cell_i%nCellsXY;                      // i%x

      
      // compute energy dispersion
      sigmaR[cub_i] += (pow(x_idx - Ecentroid[cub_i].X(), 2) + 
                        pow(y_idx - Ecentroid[cub_i].Y(), 2))*E;
      sigmaZ[cub_i] +=  pow(z_idx - Ecentroid[cub_i].Z(), 2)*E;
    }

    // save photon counts to file
    if(!primary_only){
      outfile.open(outputFilePath + name + "_" + to_string(i) + ".dat", std::ios::binary);
      if (!outfile) {
        cerr << "Error opening file for writing during event " << i << "!" << endl;
        return;
      }
    }

    // Write data to file
    for(int i_cub = 0; i_cub < nCublets; i_cub++) {

      // check if energy has been released in the cublet, otherwise skip
      if((primary_only && (i_cub != primary_peak_cub)) || !(Etot[i_cub] > 0)) continue;

      // photon_matrix data
      for(int i_t = 0; i_t < timesteps; i_t++){
        for(int i_s = 0; i_s < n_sensors; i_s++){
          int entry = photon_matrix[i_cub][i_t][i_s];
          if(entry != 0){
            outfile.write(reinterpret_cast<char*>(&i_t), sizeof(i_t));
            outfile.write(reinterpret_cast<char*>(&i_s), sizeof(i_s));
            outfile.write(reinterpret_cast<char*>(&entry), sizeof(entry));
          }
        }
      }

      // print stop bits
      int stop = 2147483647;
      outfile.write(reinterpret_cast<char*>(&stop), sizeof(stop));

      // cublet index
      outfile.write(reinterpret_cast<char*>(&i_cub), sizeof(i_cub));

      // total energy
      outfile.write(reinterpret_cast<char*>(&Etot[i_cub]), sizeof(Etot[i_cub]));
      
      // energy centroid
      outfile.write(reinterpret_cast<char*>(&Ecentroid[i_cub].X()), sizeof(Ecentroid[i_cub].X()));
      outfile.write(reinterpret_cast<char*>(&Ecentroid[i_cub].Y()), sizeof(Ecentroid[i_cub].Y()));
      outfile.write(reinterpret_cast<char*>(&Ecentroid[i_cub].Z()), sizeof(Ecentroid[i_cub].Z()));

      // energy dispersion
      sigmaR[i_cub] /= Etot[i_cub];
      sigmaZ[i_cub] /= Etot[i_cub];
      outfile.write(reinterpret_cast<char*>(&sigmaR[i_cub]), sizeof(sigmaR[i_cub]));
      outfile.write(reinterpret_cast<char*>(&sigmaZ[i_cub]), sizeof(sigmaZ[i_cub]));

      // number of interactions
      outfile.write(reinterpret_cast<char*>(&Nint[i_cub]), sizeof(Nint[i_cub]));

      // particle info
      outfile.write(reinterpret_cast<char*>(&p[i_cub]), sizeof(p[i_cub]));

      // primary vertex
      if(!primary_only){
        int is_primary = (i_cub == primary_peak_cub);
        outfile.write(reinterpret_cast<char*>(&is_primary), sizeof(is_primary));
      }

      if(verbose > 1){
        cout << endl;
        cout << "\ti_cub: " << i_cub << " - Etot: " << Etot[i_cub] << " - dEmax: " << dEmax[i_cub] 
             << " - pdg_max: " << pdg_max[i_cub] << " - classification: " << p[i_cub] << endl;
      }
    }

    // Close the file
    if(!primary_only){
      outfile.close();
    }
  }

  // Close the file
  if(primary_only){
    outfile.close();
  }

  file->Close();

  // Stop measuring time
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  if(verbose) {
    cout << "Finished processing file. Time taken: " << duration.count() << " secs. " << std::endl;
  }
  
  return;
}




double onAxis_SolidAngle(double a, double b, double d) {
  double alpha = a/(2*d);
  double beta  = b/(2*d);
  return 4*TMath::ASin(alpha*beta/TMath::Sqrt((1+alpha*alpha)*(1+beta*beta))); 
}

double offAxis_SolidAngle(double A, double B, double a, double b, double d) {
  double sign_A = 1;
  if(A < 0) {
    sign_A = -1;
    A *= -1;
  }
  double sign_B = 1;
  if(B < 0) {
    sign_B = -1;
    B *= -1;
  }
  double omega1 = onAxis_SolidAngle(2*(a+sign_A*A), 2*(b+sign_B*B), d);
  double omega2 = onAxis_SolidAngle(2*A,            2*(b+sign_B*B), d);
  double omega3 = onAxis_SolidAngle(2*(a+sign_A*A), 2*B,            d);
  double omega4 = onAxis_SolidAngle(2*A,            2*B,            d);

  double omega = (omega1 - sign_A*omega2 - sign_B*omega3 + sign_A*sign_B*omega4)/4;

  return omega;
}

// assumes sensors on upper xz plane
vector<vector<vector<vector<double>>>> create_matrices(double cellSizeX, double cellSizeY, double cellSizeZ,
                                                       int nCellsX, int nCellsY, int nCellsZ) {

  // light speed
  double n = 2.2; // PWO refractive index
  double c0 = 299.792458/n; // mm/ns
  double c = c0/n;

  vector<vector<vector<double>>> angle_matrix(nCellsX, vector<vector<double>>(nCellsY, vector<double>(nCellsZ, 0)));
  vector<vector<vector<double>>>  time_matrix(nCellsX, vector<vector<double>>(nCellsY, vector<double>(nCellsZ, 0)));
  
  // loop over y
  for(int j = 0; j < nCellsY; j++) {
    TVector3 start_point(0, j*cellSizeY, 0);

    // loop over z
  	for(int k = 0; k < nCellsZ; k++) {
      // loop over x 
      for(int i = 0; i < nCellsX; i++) {
        TVector3 end_point(i*cellSizeX, (nCellsY-0.5)*cellSizeY, k*cellSizeZ);

        double d = end_point.Y() - start_point.Y();
        double A = end_point.X() - cellSizeX/2;
        double B = end_point.Z() - cellSizeZ/2;
        angle_matrix[i][j][k] = offAxis_SolidAngle(A, B, cellSizeX, cellSizeZ, d);
        time_matrix[i][j][k] = (end_point - start_point).Mag()/c;
      }
    }      
  } 
  vector<vector<vector<vector<double>>>> result;
  result.push_back(angle_matrix);
  result.push_back(time_matrix);

  return result;
}


int main(int argc, char* argv[]) {

  string fileName = "";
  string outputFilePath = "./";
  int verbose = 0;
  bool primary_only = false;
  int max_event = 1000;
  for (int i = 1; i < argc; i++) {
    std::string flag(argv[i]);

    if (flag.find("fileName") != string::npos) {
      fileName = flag.substr(11);
    }
    else if (flag=="-f") {
      i += 1;
      fileName = argv[i];
    }

    else if (flag.find("output") != string::npos) {
      outputFilePath = flag.substr(9);
    }
    else if (flag=="-o") {
      i += 1;
      outputFilePath = argv[i];
    }

    else if (flag.find("verbose") != string::npos) {
      verbose = std::stoi(flag.substr(10));
    }
    else if (flag=="-v") {
      i += 1;
      verbose = std::stoi(argv[i]);
    }

    else if (flag.find("primary_only") != string::npos) {
      primary_only = true;
    }
    else if (flag=="-po") {
      primary_only = true;
    }

    else if (flag.find("max_event") != string::npos) {
      max_event = std::stoi(flag.substr(12));
    }
    else if (flag=="-e") {
      i += 1;
      max_event = std::stoi(argv[i]);
    }
  }

  if(fileName == "") {
    cout << "Missing file to be analyzed. Provide either with '-f <filename>' or '--fileName=<filename>" << endl;
    return 0;
  }

  if(outputFilePath[outputFilePath.size()-1] != '/'){
    outputFilePath += '/';
  }

  cout << "Computing matrices..." << endl;
  vector<vector<vector<vector<double>>>> matrices = create_matrices(cellSizeXY, cellSizeXY, cellSizeZ,
                                                                    nCellsXY,   nCellsXY,   nCellsZ);
  cout << "Timing and solid angle matrices computed.\n"
       << "\n---------------------------------------\n\n"
       << "Analyzing file " << fileName << ":" << endl;



  genPhotonTree(fileName, "outputTree", outputFilePath, matrices[0],
                matrices[1], verbose, primary_only, max_event);

	cout << "File processing completed." << endl;

  return 0;
}