#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <tuple>
#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TVector3.h"
#include "TRandom3.h"

using namespace std;

enum Particle {
  unclassified = 0,
  proton = 1,
  kaon = 2,
  pion = 3,
  other = 4
};

// set useful constant values

// detector geometry
array<int, 3> nCublets = {10, 10, 10};
int TotCublets = nCublets[0]*nCublets[1]*nCublets[2];
array<double, 3> cubletSize = {30, 30, 120}; //mm
array<double, 3> startCublet = {
                  -cubletSize[0] * (1.0*(nCublets[0])/2), // -150 mm
                  -cubletSize[1] * (1.0*(nCublets[1])/2), // -150 mm
                  -cubletSize[2] * (1.0*(nCublets[2])/2)  // -600 mm
                  }; // position of the left side of the first cublet in the calorimeter
array<int, 3> nCells = {10, 10, 10}; 
array<double, 3> cellSize = {3, 3, 12}; //mm
array<double, 3> startCell = {
                    -cellSize[0] * (1.0*(nCells[0])/2), // -15 mm
                    -cellSize[1] * (1.0*(nCells[1])/2), // -15 mm
                    -cellSize[2] * (1.0*(nCells[2])/2)  // -6 mm
                  }; // position of the left side of the first cell (relative to the centre of the cublet)
//light characteristics
double deltaE_vtx_thr = -50e3; // MeV, threshold of energy loss to be considered
                               //      primary vertex of the event
int n_sensors = nCells[0]*nCells[2]; // along xz plane
double lightyield = 200; // ph/MeV 
double max_t = 20; // ns
double dt = 0.2; // ns
int timesteps = max_t/dt;


inline array<int, 3> SingleIndexTo3DIndexes(int idx, array<int, 3> nUnits){
  array<int, 3> indexes = {
                     idx%nUnits[0],                        // i%x
                    (idx%(nUnits[0]*nUnits[1]))/nUnits[0], // (i%(x*y))/x
                     idx/(nUnits[0]*nUnits[1]),            // i/(x*y)
                  };
  return indexes;
}

// convert cublet and cell indexes in 3D coordinates
array<double, 3> IndexesToCoordinates(int cublet_idx, int cell_idx){
  array<int, 3> cublet_idxs = SingleIndexTo3DIndexes(cublet_idx, nCublets);
  array<int, 3> cell_idxs   = SingleIndexTo3DIndexes(cell_idx,   nCells);
  array<double, 3> coordinates;
  for(int i = 0; i < nCublets.size(); i++){
    coordinates[i] = startCublet[i] + (cublet_idxs[i]+0.5)*cubletSize[i] +
                     startCell[i]   + (cell_idxs[i]+0.5)  *cellSize[i];
  }
  return coordinates;
}

array<int, 4> CoordinatesToIndexes(array<double, 3> coordinates){
  array<int, 3> cublet_idxs, cell_idxs;
  for(int i = 0; i < coordinates.size(); i++){
    cublet_idxs[i] = (coordinates[i] -  startCublet[i])/cubletSize[i]; 
    cell_idxs[i]   = (coordinates[i] - (startCublet[i]+cublet_idxs[i]*cubletSize[i]))/cellSize[i]; 
  }
  int cublet_idx = cublet_idxs[0] +
                   cublet_idxs[1] * nCublets[0] +
                   cublet_idxs[2] * nCublets[0]*nCublets[1];
  return {cublet_idx, cell_idxs[0], cell_idxs[1], cell_idxs[2]}; 
}


// convert cublet and cell indexes to new coordinates system:
array<int, 4> CoordinatesShift(int cublet_idx, int cell_idx, array<double, 2> shift){
  array<double, 3> coordinates = IndexesToCoordinates(cublet_idx, cell_idx);
  for(int i = 0; i < shift.size(); i++){
    coordinates[i] += shift[i];
  }
  array<int, 4> results = CoordinatesToIndexes(coordinates);
  return results;
}



vector<float> read_matrices(string filename){
    // Step 1: Read the shape from the text file
    ifstream shape_file("shape.txt");
    vector<size_t> shape;
    if (shape_file.is_open()) {
      string line;
      getline(shape_file, line);
      istringstream iss(line);
      size_t dim;
      while (iss >> dim) {
          shape.push_back(dim);
      }
    }
    else {
      cerr << "Failed to open shape.txt" << endl;
    }

    // Step 2: Calculate the total number of elements
    size_t total_elements = 1;
    for(size_t dim : shape){
      total_elements *= dim;
    }

    // Step 3: Read the binary data
    ifstream binary_file(filename, ios::binary);
    if (!binary_file.is_open()) {
      cerr << "Failed to open tensor.bin" << endl;
    }

    vector<float> data(total_elements);  // Assuming float32 data type
    binary_file.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(float));

    return data;
}

int total_reflections(int n){
  vector<int> extra_points;
  for(int i = 0; i < n+1; i++){
    switch(i){
      case 0:
        extra_points.push_back(1);
        break;
      case 1:
        extra_points.push_back(5);
        break;
      default:
        extra_points.push_back(4*(2*i-1));
    }
  }

  int total_points = 0;
  for(int i = 0; i < extra_points.size(); i++){
    total_points += extra_points[i];
  }

  return total_points;
}





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
                   vector<float>& emission_matrix, int max_N,
                   TRandom3 rng,
                   int verbose=0, bool primary_only=true, int max_event=1000) {

  auto start_time = std::chrono::high_resolution_clock::now();

  int total_points = total_reflections(max_N);
  vector<int> shape{nCells[0], nCells[1], nCells[2], total_reflections(5), nCells[0], nCells[2], 2};
  int dims = shape.size();

  size_t name_start = filename.find_last_of('/');
  size_t name_end   = filename.find_last_of('.');
  string name = filename.substr(name_start+1, name_end-name_start-1);

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

  
  vector<vector<vector<int>>> photon_matrix(TotCublets,  vector<vector<int>>(
                                            timesteps, vector<int>(
                                            n_sensors, 0)));
  vector<double> dEmax(TotCublets, 0.0);                   // maximum energy diff between step beginning and end...
  vector<double> Etot(TotCublets, 0.0);                    // total energy released...
  vector<TVector3> Ecentroid(TotCublets, TVector3(0,0,0)); // (weighted) centroid of energy depositions...
  vector<TVector3> sigmaE(TotCublets,    TVector3(0,0,0)); // (weighted) energy dispersion along x, y and z...
  vector<Particle> p(TotCublets, unclassified);            // particle classification...
  vector<int> Nint(TotCublets, 0);                         // number of interactions...
  vector<int> pdg_max(TotCublets, 0);                      // pdg encoding of primary particle...
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
    fill(sigmaE.begin(),    sigmaE.end(),    TVector3(0,0,0));
    fill(p.begin(),         p.end(),         unclassified);
    fill(Nint.begin(),      Nint.end(),      0);
    fill(pdg_max.begin(),   pdg_max.end(),   0);

    // primary vertex identification variables
    double dE_primary = 0;
    int primary_peak_cub = -1;

//cout << "CREATE SHIFT" << endl;
    // generate random shift
    array<double, 2> shift = {rng.Uniform(0, cubletSize[0]), rng.Uniform(0, cubletSize[1])};  //{0.5*cubletSize[0], 0.5*cubletSize[1]}; 
//cout << "CREATED SHIFT" << endl;
    
    // loop over interaction per event
    for (int j = 0; j < n_int; j++) {

      // check if energy has been deposited before max time
      double E  = (*edep)[j];
      double t0 = (*glob_t)[j];
      double dE = (*deltae)[j];
      if (E > 0 && t0 < max_t) {

        int old_cub_i = (*cublet_idx)[j];
        int old_cell_i = (*cell_idx)[j];
        // shift cublet and cell indexes
//cout << "PROVA" << endl;
        auto new_coordinates = CoordinatesShift(old_cub_i, old_cell_i, shift);
//cout << "PROVA2" << endl;
        int cub_i = new_coordinates[0];
        int x_idx = new_coordinates[1];
        int y_idx = new_coordinates[2];
        int z_idx = new_coordinates[3];
        if(cub_i < 0 || cub_i >= TotCublets) continue;   
//cout << cub_i << endl;
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
//        cout << "QUI 1" << endl;

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

        //int cell_i = (*cell_idx)[j];
        //int z_idx =  cell_i/(nCellsXY*nCellsXY);           // i/(x*y)
        //int y_idx = (cell_i%(nCellsXY*nCellsXY))/nCellsXY; // (i%(x*y))/x
        //int x_idx =  cell_i%nCellsXY;                      // i%x

        // update centroid
        Ecentroid[cub_i].SetXYZ(Ecentroid[cub_i].X() + x_idx*E,
                                Ecentroid[cub_i].Y() + y_idx*E,
                                Ecentroid[cub_i].Z() + z_idx*E);
//cout << "QUI 2" << endl;
        
        // compute photon arriving to sensors
        double ph_emitted = E*lightyield;

        // loop over sensors 
        int chunk_size = total_points*shape[dims-3]*shape[dims-2]*shape[dims-1];
        vector<float> cached_chunk(chunk_size);
        int chunk_start = z_idx * shape[dims-4]*shape[dims-3]*shape[dims-2]*shape[dims-1] +
                          y_idx * shape[dims-5]*shape[dims-4]*shape[dims-3]*shape[dims-2]*shape[dims-1] +
                          x_idx * shape[dims-6]*shape[dims-5]*shape[dims-4]*shape[dims-3]*shape[dims-2]*shape[dims-1];
//cout << "Chunk:  " << chunk_start << "   " << chunk_size << "   " << &(*emission_matrix.begin()) << endl
//     << "        " << x_idx << "   " << y_idx << "   " << z_idx << endl;
        std::copy(emission_matrix.begin() + chunk_start,
                  emission_matrix.begin() + chunk_start + chunk_size,
                  cached_chunk.begin());
        for(int n = 0; n < total_points; n++){
          for (int i_sx = 0; i_sx < nCells[0]; i_sx++) {
            for (int i_sz = 0; i_sz < nCells[2]; i_sz++) {
              int sensor_i = i_sz*nCells[0]+i_sx;
              int idx = i_sz * shape[dims-1] +
                        i_sx * shape[dims-2]*shape[dims-1] +
                        n    * shape[dims-3]*shape[dims-2]*shape[dims-1];
              int n_photon = round(ph_emitted*cached_chunk[idx]);
              double time = (t0+cached_chunk[idx+1]);
              int step = time/dt;
//              cout << "Matrix:  " << idx << "   " << cached_chunk[idx] << "   " << cached_chunk[idx+1] << endl 
//                   << "  Photons:  " << ph_emitted << "   " << n_photon << endl
//                   << "  Time:  " << t0 << "   " << time << "   " << step << endl;
              if(time < max_t) {
                photon_matrix[cub_i][step][sensor_i] += n_photon;
              }
//              cout << cub_i << "   " << step << "   " << sensor_i << endl;
            }
          }
        }
//cout << "testergs" << endl;
      }
    }


//cout << "QUI 3" << endl;

    // correct centroid estimation
    for(int i_cub = 0; i_cub < TotCublets; i_cub++) {
      if(!(Etot[i_cub] > 0)) continue;
      Ecentroid[i_cub].SetXYZ(Ecentroid[i_cub].X()/Etot[i_cub],
                              Ecentroid[i_cub].Y()/Etot[i_cub],
                              Ecentroid[i_cub].Z()/Etot[i_cub]);
    }
//cout << "QUI 4" << endl;


    // compute energy dispersions
    for (int j = 0; j < n_int; j++) {
      int old_cub_i = (*cublet_idx)[j];
      int old_cell_i = (*cell_idx)[j];
      // shift cublet and cell indexes
      auto new_coordinates = CoordinatesShift(old_cub_i, old_cell_i, shift);
      int cub_i = new_coordinates[0];
      int x_idx = new_coordinates[1];
      int y_idx = new_coordinates[2];
      int z_idx = new_coordinates[3];

      if(cub_i < 0 || cub_i >= TotCublets) continue;  

      // check if energy has been released in the cublet, otherwise skip
      if((primary_only && (cub_i != primary_peak_cub)) || !(Etot[cub_i] > 0)) continue;
      
      double E  = (*edep)[j];
      
      // update energy dispersion vector
      sigmaE[cub_i].SetXYZ(sigmaE[cub_i].X() + pow(x_idx - Ecentroid[cub_i].X(), 2)*E,
                           sigmaE[cub_i].Y() + pow(y_idx - Ecentroid[cub_i].Y(), 2)*E,
                           sigmaE[cub_i].Z() + pow(z_idx - Ecentroid[cub_i].Z(), 2)*E);
    }
//cout << "QUI 5" << endl;


    // save photon counts to file
    if(!primary_only){
      outfile.open(outputFilePath + name + "_" + to_string(i) + ".dat", std::ios::binary);
      if (!outfile) {
        cerr << "Error opening file for writing during event " << i << "!" << endl;
        return;
      }
    }

    // Write data to file
    for(int i_cub = 0; i_cub < TotCublets; i_cub++) {

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
      double x = Ecentroid[i_cub].X();
      double y = Ecentroid[i_cub].Y();
      double z = Ecentroid[i_cub].Z();
      outfile.write(reinterpret_cast<char*>(&x), sizeof(x));
      outfile.write(reinterpret_cast<char*>(&y), sizeof(y));
      outfile.write(reinterpret_cast<char*>(&z), sizeof(z));

      // energy dispersion
      double sX = sigmaE[i_cub].X()/Etot[i_cub];
      double sY = sigmaE[i_cub].Y()/Etot[i_cub];
      double sZ = sigmaE[i_cub].Z()/Etot[i_cub];
      outfile.write(reinterpret_cast<char*>(&sX), sizeof(sX));
      outfile.write(reinterpret_cast<char*>(&sY), sizeof(sY));
      outfile.write(reinterpret_cast<char*>(&sZ), sizeof(sZ));

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
  int reflections = 0;
  int seed = -1;
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
    
    else if (flag.find("reflections") != string::npos) {
      reflections = std::stoi(flag.substr(14));
    }
    else if (flag=="-r") {
      i += 1;
      reflections = std::stoi(argv[i]);
    }

    else if (flag.find("seed") != string::npos) {
      reflections = std::stoi(flag.substr(7));
    }
    else if (flag=="-s") {
      i += 1;
      reflections = std::stoi(argv[i]);
    }
  }

  if(fileName == "") {
    cout << "Missing file to be analyzed. Provide either with '-f <filename>' or '--fileName=<filename>" << endl;
    return 0;
  }

  if(outputFilePath[outputFilePath.size()-1] != '/'){
    outputFilePath += '/';
  }

  /*
  if(reflections == 0){
    cout << "Computing matrices..." << endl;
    vector<vector<vector<vector<double>>>> matrices = create_matrices(cellSizeXY, cellSizeXY, cellSizeZ,
                                                                      nCellsXY,   nCellsXY,   nCellsZ);
  }
  */

  cout << "Reading matrices..." << endl;
  vector<float> emission_matrix = read_matrices("emission_matrix.bin");

  cout << "Timing and solid angle matrices computed.\n"
       << "\n---------------------------------------\n\n"
       << "Analyzing file " << fileName << ":" << endl;

  TRandom3 rng;
  if(seed != -1){
    rng.SetSeed(seed);
  }

  genPhotonTree(fileName, "outputTree", outputFilePath, emission_matrix,
                reflections, rng, verbose, primary_only, max_event);

	cout << "File processing completed." << endl;

  return 0;
}