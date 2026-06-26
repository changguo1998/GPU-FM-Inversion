#include "mt_utils.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

// Test binary: takes strike, dip, rake in DEGREES from command line,
// prints CSV header + one row of MomentTensor values to stdout.
//
// Usage: ./test_mt_to_csv <strike_deg> <dip_deg> <rake_deg>

static constexpr double DEG2RAD = M_PI / 180.0;

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <strike_deg> <dip_deg> <rake_deg>" << std::endl;
        return 1;
    }

    double strike = std::atof(argv[1]) * DEG2RAD;
    double dip = std::atof(argv[2]) * DEG2RAD;
    double rake = std::atof(argv[3]) * DEG2RAD;

    MomentTensor mt = sdr_to_mt(strike, dip, rake);

    // CSV header + data row.
    std::cout << std::setprecision(17);
    std::cout << "strike,dip,rake,Mxx,Myy,Mzz,Mxy,Mxz,Myz" << "\n";
    std::cout << std::atof(argv[1]) << "," << std::atof(argv[2]) << "," << std::atof(argv[3]) << ","
              << mt.Mxx << "," << mt.Myy << "," << mt.Mzz << "," << mt.Mxy << "," << mt.Mxz << ","
              << mt.Myz << "\n";

    return 0;
}
