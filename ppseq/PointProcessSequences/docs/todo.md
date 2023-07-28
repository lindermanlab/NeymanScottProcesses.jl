# TODO


REFACTORS
* [ ] refactor sampling
* [ ] refactor `distributed.jl`
* [ ] refactor `distributed_gibbs.jl`


TESTS
* [ ] unit tests


MINOR
* [ ] should `reset!` also reset the sampled variables?
* [ ] generalize `event_list_summary` to use named tuples
* [ ] retype `latent_event_hist` and `globals_hist` in `gibbs_sample!`
* [ ] change second arg of `set_posterior!` to be an event, not an integer
* [ ] change float64 types to reals or# abstract floats
* [ ] modify add/remove datapoint name and args


PERFORMANCE
* [ ] static arrays


DONE
* [X] define abstract structs
* [X] write generic methods
* [X] outline specific methods
* [X] one directory for each model (and base model)
* [X] choose common filenames
* [X] setup test suites
* [X] time/spatial parameter
* [X] event rate parameter
* [X] refactor adding / removing events
* [X] refactor `gibbs.jl`