using Pkg

# Install current environment
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Install synthetic experiment environments
Pkg.activate(joinpath(@__DIR__, "experiments", "dpm_limit"))
Pkg.instantiate()

# Install embassy experiment environments
Pkg.activate(joinpath(@__DIR__, "experiments", "embassy"))
Pkg.instantiate()

# Install neural experiment environments
Pkg.activate(joinpath(@__DIR__, "ppseq"))
Pkg.instantiate()
