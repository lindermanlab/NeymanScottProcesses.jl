using NeymanScottProcesses
using Plots

bounds = (1.0, 1.0)

K = 1.0  # event rate
Ak = specify_gamma(20.0, 3.0)  # event amplitude
A0 = specify_gamma(20.0, 3.0)  # background amplitude
Ψ = 1e-3 * [1.0 0; 0 1]  # covariance scale
ν = 5.0  # covariance degrees of freedom

priors = GaussianPriors(K, Ak, A0, Ψ, ν)
model = GaussianNeymanScottModel(bounds, priors)

data, assignments, events = sample(model; resample_latents=true)

plot(data, assignments)