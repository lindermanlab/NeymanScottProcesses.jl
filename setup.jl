using Pkg

# Install current environment
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Install experiment environments
Pkg.activate(joinpath(@__DIR__, "experiments", "dpm_limit"))
Pkg.instantiate()

# Install experiment environments
Pkg.activate(joinpath(@__DIR__, "experiments", "embassy"))
Pkg.instantiate()