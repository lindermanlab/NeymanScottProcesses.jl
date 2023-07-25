# PointProcessSequences.jl


### Installation Instructions

First, install [Julia](https://julialang.org/downloads/) version 1.3 or later.

Then, make `PointProcessSequences` your working directory and run

```
$ julia
```

To install the package, hit `]` to enter `pkg` mode and run

```
julia> ]
(v1.3) pkg> develop .
```

Finally, to run the package, press backspace to exit `pkg` mode and run

```
julia> using PointProcessSequences
julia> include("examples/songbird.jl")
julia>
julia> spikes, max_time, num_neurons = load_songbird_data()
julia> model = load_songbird_model(
            max_time,
            num_neurons
        )
julia> results = gibbs_sample!(
            model,
            spikes,
            fill(-1, length(spikes)),  # initial assignments, all background
            1000,  # number of gibbs samples
            0,  # split merge moves
            1.0,  # split merge window
            10  # save results every 10 samples
        )
```




