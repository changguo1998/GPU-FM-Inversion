module MT

"""
    sdr_to_mt(strike::Float64, dip::Float64, rake::Float64) -> Vector{Float64}

Convert strike/dip/rake (degrees) to a 6-component moment tensor
in NED coordinate system: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`.

# Arguments
- `strike`: strike angle in degrees, ∈ [0, 360)
- `dip`: dip angle in degrees, ∈ [0, 90]
- `rake`: rake angle in degrees, ∈ [-90, 90]

# Returns
- 6-element `Vector{Float64}`: `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`

# Example
```julia
julia> sdr_to_mt(0.0, 90.0, 0.0)
6-element Vector{Float64}:
 0.0
 0.0
 0.0
 1.0
 0.0
 0.0
```
"""
function sdr_to_mt(strike::Float64, dip::Float64, rake::Float64)::Vector{Float64}
    s = deg2rad(strike)
    d = deg2rad(dip)
    r = deg2rad(rake)

    sin_s = sin(s)
    cos_s = cos(s)
    sin_d = sin(d)
    cos_d = cos(d)
    sin_r = sin(r)
    cos_r = cos(r)

    sin_2s = sin(2s)
    cos_2s = cos(2s)
    sin_2d = sin(2d)
    cos_2d = cos(2d)

    Mxx = -(sin_2s * sin_d * cos_r + sin_s^2 * sin_2d * sin_r)
    Myy = sin_2s * sin_d * cos_r - cos_s^2 * sin_2d * sin_r
    Mzz = sin_2d * sin_r
    Mxy = cos_2s * sin_d * cos_r + 0.5 * sin_2s * sin_2d * sin_r
    Mxz = -(cos_s * cos_d * cos_r + sin_s * cos_2d * sin_r)
    Myz = -(sin_s * cos_d * cos_r - cos_s * cos_2d * sin_r)

    return [Mxx, Myy, Mzz, Mxy, Mxz, Myz]
end

"""
    sdr_to_mt_batch(strikes::Vector{Float64}, dips::Vector{Float64}, rakes::Vector{Float64}) -> Matrix{Float64}

Convert multiple strike/dip/rake combinations (degrees) to moment tensors.

# Arguments
- `strikes`: vector of strike angles in degrees, length N
- `dips`: vector of dip angles in degrees, length N
- `rakes`: vector of rake angles in degrees, length N

# Returns
- `Matrix{Float64}` of shape `[6, N]` where each column is `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]`
"""
function sdr_to_mt_batch(
    strikes::Vector{Float64},
    dips::Vector{Float64},
    rakes::Vector{Float64},
)::Matrix{Float64}
    n = length(strikes)
    @assert length(dips) == n && length(rakes) == n "All input vectors must have the same length"
    result = Matrix{Float64}(undef, 6, n)
    for i in 1:n
        result[:, i] .= sdr_to_mt(strikes[i], dips[i], rakes[i])
    end
    return result
end

export sdr_to_mt, sdr_to_mt_batch

end # module
