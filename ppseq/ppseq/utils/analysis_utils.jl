function rand_2d_projection(M::AbstractMatrix)
    n = size(M, 1)
    P = randn(n, 2)
    P[:, 1] ./= norm(view(P, :, 1))
    P[:, 2] .-= dot(view(P, :, 1), view(P, :, 2)) .* P[:, 1]
    P[:, 2] ./= norm(view(P, :, 2))
    return P' * M
end

function pca_2d_projection(M::AbstractMatrix)
    U, S, Vt = svd(M .- mean(M, dims=2))
    U[:, 1:2]' * M
end

function get_matchings(X::Array{Float64, 3}, template::Matrix{Float64})

    m, n, N = size(X)

    neg_tmp = -1 * template
    new_tmp = zeros(size(template))

    for iterations in 1:3
        println("matching, iter = ", iterations)
        for i = 1:N
            Xi = X[:, :, i]
            matching = Hungarian.munkres(Xi' * neg_tmp)
            new_tmp .+= (Xi * matching)
        end
        neg_tmp .= new_tmp * (-1 / N)
    end

    return [Hungarian.munkres(X[:, :, i]' * neg_tmp) for i = 1:N]

end

function neuron_response_confidence_intervals(
        amplitudes::Array{Float64, 3},
        offsets::Array{Float64, 3},
        widths::Array{Float64, 3},
    )

    nothing
end


"""Sort neurons by preferred sequence type and offset."""
function sortperm_neurons(globals::PointProcessSequences.SeqGlobals; thres=0.2)

    resp_props = exp.(globals.neuron_response_log_proportions)
    offsets = globals.neuron_response_offsets
    peak_resp = dropdims(maximum(resp_props, dims=2), dims=2)

    has_response = peak_resp .> quantile(peak_resp, thres) 
    preferred_type = [idx[2] for idx in argmax(resp_props, dims=2)]
    preferred_delay = [offsets[n, r] for (n, r) in enumerate(preferred_type)]

    Z = collect(zip(has_response, preferred_type, preferred_delay))
    return sortperm(sortperm(Z))
end


"""Spike x Spike co-occupancy probability matrix over assignments."""
function spike_co_occupancy(assignment_hist::Matrix{Int64})

    num_spikes, num_samples = size(assignment_hist)

    C = zeros(Int, num_spikes, num_spikes)

    for s = 1:num_samples

        for i = 1:num_spikes

            if assignment_hist[i, s] == -1
                continue
            end

            for j = (i + 1):num_spikes
                if assignment_hist[i, s] == assignment_hist[j, s]
                    C[i, j] += 1
                    C[j, i] += 1
                end
            end

        end
    end

    P = C / num_samples
    P[diagind(P)] .= 1.0

    i2 = Clustering.hclust(1.0 .- P, branchorder=:optimal).order

    return P, P[i2, :][:, i2]
end
