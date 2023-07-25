import Random
import PyPlot: plt

using LinearAlgebra

Random.seed!(1)

# Specify parameters
bounds = (1.0, 1.0)
area = bounds[1] * bounds[2]
max_radius = 5.0

event_rate = 5.0 # events ≈ 5
event_amplitude = specify_gamma(30.0, 30.0^2)
bkgd_amplitude = specify_gamma(10.0, 1e-8)  
covariance_scale = [1.0 0.0; 0.0 1.0] * 1e-3
covariance_df = 10.0


# Generate empty model
model = GaussianNeymanScottModel(
    bounds,
    max_radius,

    event_rate,
    event_amplitude,
    bkgd_amplitude,
    covariance_scale,
    covariance_df
)

#background_prob = background_assignment_prob(model)
#new_cluster_prob = new_cluster_assignment_prob(model)

#@show background_prob
#@show new_cluster_prob

temp = 1:1:100
prob_ratios = log.(
    prob_ratio_vs_bkgd_temp(30.0, 30.0^2, event_rate, 10.0, area, temp)
)


plt.plot(temp, prob_ratios)
plt.xlabel("temperature")
plt.ylabel("log (bkgd prob / new event prob)")