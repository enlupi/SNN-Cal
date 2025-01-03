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
