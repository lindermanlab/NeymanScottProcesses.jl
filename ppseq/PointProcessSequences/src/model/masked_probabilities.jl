abstract type AbstractMask end

struct SeqMask <: AbstractMask
    neuron::Int64
    window::Tuple{Float64, Float64}
end

struct GaussianMask <: AbstractMask
    center::Tuple{Float64, Float64}
    radius::Float64
end

struct GaussianComplementMask <: AbstractMask
    masks::Vector{GaussianMask}
    bounds::Tuple{Float64, Float64}
end

struct CablesMask <: AbstractMask
    window::Tuple{Float64, Float64}
end

start(mask::SeqMask) = mask.window[1]
start(mask::CablesMask) = mask.window[1]

stop(mask::SeqMask) = mask.window[2]
stop(mask::CablesMask) = mask.window[2]

in(x::Spike, mask::SeqMask) = (
    (start(mask) < x.timestamp < stop(mask))
    && (x.neuron == mask.neuron)
)
in(x::Point, mask::GaussianMask) = (norm(x.position .- mask.center) < mask.radius)
in(x::Cable, mask::CablesMask) = (start(mask) < x.position < stop(mask))
in(x::Point, comp_mask::GaussianComplementMask) = !(x in comp_mask.masks)

volume(mask::SeqMask) = mask.window[1] - mask.window[2]
volume(mask::GaussianMask) = π * mask.radius^2
volume(mask::CablesMask) = mask.window[1] - mask.window[2]
function volume(complement_mask::GaussianComplementMask; num_samples=1000)
    num_in_complement_mask = 0
    (a1, a2) = complement_mask.bounds

    for i in 1:num_samples
        x = Point([rand()*a1, rand()*a2])
        if x in complement_mask
            num_in_complement_mask += 1
        end
    end

    return (a1*a2) * (num_in_complement_mask / num_samples)
end


integrated_bkgd_intensity(model::NeymanScottModel, mask::AbstractMask) =
    bkgd_rate(model.globals) * volume(mask)

integrated_bkgd_intensity(model::PPSeq, mask::SeqMask) = (
    bkgd_rate(model.globals) 
    * exp(model.globals.bkgd_log_proportions[mask.neuron]) 
    * volume(mask)
)

function integrated_event_intensity(model::PPSeq, event::SeqEvent,  mask::SeqMask)
    globals = model.globals
    
    w = warps[event.sampled_warp]
    r = event.sampled_type
    n = mask.neuron

    g = Normal(
        w * globals.offsets[n, r] + event.sampled_timestamp,
        w * sqrt(globals.widths[n, r]) 
    )

    return (
        event.sampled_amplitude
        * exp(globals.neuron_response_log_proportions[n, r])
        * (cdf(g, stop(mask)) - cdf(g, start(mask)))
    )
end

function integrated_event_intensity(
    model::CablesModel, event::CableCluster, mask::CablesMask
)
    g = Normal(event.sampled_position, sqrt(event.sampled_variance))

    return (
        event.sampled_amplitude
        * (cdf(g, stop(mask)) - cdf(g, start(mask)))
    )
end

function integrated_event_intensity(
    model::GaussianNeymanScottModel, event::Cluster, mask::GaussianMask;
    num_samples = 1000
)
    g = MultivariateNormal(event.sampled_position, event.sampled_covariance)


    num_in_mask = 0

    for _ = 1:num_samples
        x = Point(rand(g))

        if x in mask
            num_in_mask += 1
        end
    end

    # Compute fraction in the mask
    prob_in_mask = sum(num_in_mask) / num_samples

    return prob_in_mask * event.sampled_amplitude
end


function integrated_event_intensity(
    model::GaussianNeymanScottModel, event::Cluster, comp_mask::GaussianComplementMask;
    num_samples = 1000
)
    g = MultivariateNormal(event.sampled_position, event.sampled_covariance)

    # Draw samples
    num_in_complement_mask = 0
    for _ in 1:num_samples
        x = Point(rand(g))

        if x in comp_mask
            num_in_complement_mask += 1
        end
    end

    # Compute fraction in the mask
    prob_in_mask = num_in_complement_mask / num_samples

    return prob_in_mask * event.sampled_amplitude
end
    
# Each mask is a tuple, (n, (t0, t1)) indicating that the neuron
# at index n is masked over the time interval (t0, t1)
# const Mask = Tuple{Int64,Tuple{Float64,Float64}}

"""
Computes log-likelihood in masked regions.
"""
function log_like(
        model::NeymanScottModel,
        data::Vector{<: AbstractDatapoint},
        masks::Vector{<: AbstractMask}
    )
    ll = 0.0

    # == FIRST TERM == #
    # -- Sum of Poisson Process intensity at all datapoints -- #
    for x in data
        # Compute intensity.
        g = bkgd_intensity(model, x)
        for event in events(model)
            g += event_intensity(model, event, x) * amplitude(event)
        end

        # Add term to log-likelihood.
        ll += log(g)
    end

    # == SECOND TERM == #
    # -- Penalty on integrated intensity function -- #
    for mask in masks
        
        # Add contribution of background.
        ll -= integrated_bkgd_intensity(model, mask)

        # Add contribution of each latent event.
        for event in events(model)
            ll -= integrated_event_intensity(model, event, mask)
        end

    end

    return ll

end


"""
Computes log-likelihood of homogeneous poisson process
of spikes within a masked region.
"""
function homogeneous_baseline_log_like(
    data::Vector{<: AbstractDatapoint}, 
    masks::Vector{<: AbstractMask}
)
    return length(data) / sum(volume.(masks))
end


function homogeneous_baseline_log_like(
    spikes::Vector{Spike},
    masks::Vector{SeqMask}
)

    # Count number of spikes and total time observed for each neuron.
    time_per_neuron = Dict{Int64,Float64}()
    spikes_per_neuron = Dict{Int64,Int64}()

    for mask in masks
        # TODO - could add a dependency to DataStructures.jl and use a DefaultDict here.
        n = mask.neuron
        if !(n in keys(time_per_neuron))
            time_per_neuron[n] = 0.0
            spikes_per_neuron[n] = 0
        end
        time_per_neuron[n] += volume(mask)
    end

    for x in spikes
        spikes_per_neuron[x.neuron] += 1
    end

    # Compute MLE estimate for a homogeneous PP.
    neurons = keys(time_per_neuron)
    mle_rates = Dict(n => max(eps(), spikes_per_neuron[n]) / time_per_neuron[n] for n in neurons)

    # Use MLE to compute total log-likelihood.
    ll = 0.0
    for n in neurons
        ll += spikes_per_neuron[n] * log(mle_rates[n])
        ll -= time_per_neuron[n] * mle_rates[n]
    end
    return ll

end
