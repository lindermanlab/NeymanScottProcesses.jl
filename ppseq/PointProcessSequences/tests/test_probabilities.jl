using PointProcessSequences
import PointProcessSequences: add_datapoint!, add_event!, log_posterior_predictive, gibbs_update_latents!

include("./util.jl")

model_1, spikes = make_test_spikes_and_model(num_warp_values=1)
model_2, _ = make_test_spikes_and_model(num_warp_values=2)

# Copy model_1 global parameters to model_2
model_2.globals = model_1.globals

add_event!(model_1, spikes[1])
add_event!(model_2, spikes[1])

for x in spikes[2:4]
    add_datapoint!(model_1, x, 1)
    add_datapoint!(model_2, x, 1)
end

gibbs_update_latents!(model_1)
gibbs_update_latents!(model_2)

lp1 = log_posterior_predictive(model_1.sequence_events[1], spikes[4], model_1)
lp2 = log_posterior_predictive(model_2.sequence_events[1], spikes[4], model_2)


@show lp1 lp2

@assert isapprox(lp1, lp2)
