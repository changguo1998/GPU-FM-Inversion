#include "mt_utils.h"
#include <cmath>

// Verify that the formula matches Julia's sdr_to_mt() exactly.
// Julia converts degrees to radians and returns [Mxx, Myy, Mzz, Mxy, Mxz, Myz]:
//   Mxx = -(sin(2s)*sin(d)*cos(r) + sin(s)^2*sin(2d)*sin(r))
//   Myy =   sin(2s)*sin(d)*cos(r) - cos(s)^2*sin(2d)*sin(r)
//   Mzz =   sin(2d)*sin(r)
//   Mxy =   cos(2s)*sin(d)*cos(r) + 0.5*sin(2s)*sin(2d)*sin(r)
//   Mxz = -(cos(s)*cos(d)*cos(r) + sin(s)*cos(2d)*sin(r))
//   Myz = -(sin(s)*cos(d)*cos(r) - cos(s)*cos(2d)*sin(r))

MomentTensor sdr_to_mt(double strike_rad, double dip_rad, double rake_rad) {
    return sdr_to_mt_device(strike_rad, dip_rad, rake_rad);
}

MT_HOST_DEVICE MomentTensor sdr_to_mt_device(double strike_rad, double dip_rad, double rake_rad) {
    double s = strike_rad;
    double d = dip_rad;
    double r = rake_rad;

    // Precompute trig functions — avoids redundant calls.
    double sin_s = std::sin(s);
    double cos_s = std::cos(s);
    double sin_d = std::sin(d);
    double cos_d = std::cos(d);
    double sin_r = std::sin(r);
    double cos_r = std::cos(r);

    double sin_2s = std::sin(2.0 * s);
    double cos_2s = std::cos(2.0 * s);
    double sin_2d = std::sin(2.0 * d);
    double cos_2d = std::cos(2.0 * d);

    double Mxx = -(sin_2s * sin_d * cos_r + sin_s * sin_s * sin_2d * sin_r);
    double Myy = sin_2s * sin_d * cos_r - cos_s * cos_s * sin_2d * sin_r;
    double Mzz = sin_2d * sin_r;
    double Mxy = cos_2s * sin_d * cos_r + 0.5 * sin_2s * sin_2d * sin_r;
    double Mxz = -(cos_s * cos_d * cos_r + sin_s * cos_2d * sin_r);
    double Myz = -(sin_s * cos_d * cos_r - cos_s * cos_2d * sin_r);

    return {Mxx, Myy, Mzz, Mxy, Mxz, Myz};
}
