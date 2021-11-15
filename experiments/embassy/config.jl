CONFIG = Dict(
    :min_date => Date(1976, 6, 21),
    :max_date => Date(1976, 7, 31),
    :vocab_cutoff => 100,
    :percent_masked => 0.05,

    :max_cluster_radius => Inf,
    :cluster_rate => 1.0 / 30,
    :cluster_amplitude => specify_gamma(500, 10^2),
    :cluster_width => specify_inverse_gamma(2.0, (1e-4)^2),
    :background_amplitude => specify_gamma(1000, 10^2),
    :background_word_concentration => 1e8,
    :background_word_spread => 1.0,

    :seed => 1976,
    :samples_per_mask => 5,
    :masks_per_anneal => 3,
    :save_interval => 1,
    :temps => exp10.(vcat(range(6.0, 0.0, length=5))),
    #exp10.(vcat(range(6.0, 0, length=20), fill(0.0, 3))),
)