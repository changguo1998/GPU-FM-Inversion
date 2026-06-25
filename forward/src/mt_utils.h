#ifndef MT_UTILS_H
#define MT_UTILS_H

// Moment tensor double-couple SDR → 6-component conversion.
// Formulas match the legacy JuliaSourceMechanism.jl dc2ts() implementation.
// All angles in RADIANS. Uses double precision throughout.

// GPU-compatible attributes — no Kokkos dependency required.
#ifdef __CUDACC__
#define MT_HOST_DEVICE __host__ __device__
#else
#define MT_HOST_DEVICE
#endif

struct MomentTensor {
  double Mxx, Myy, Mzz, Mxy, Mxz, Myz;
};

// Host function: converts SDR (radians) → MomentTensor.
MomentTensor sdr_to_mt(double strike_rad, double dip_rad, double rake_rad);

// GPU device function (also callable from host).
MT_HOST_DEVICE MomentTensor sdr_to_mt_device(double strike_rad, double dip_rad,
                                             double rake_rad);

#endif // MT_UTILS_H