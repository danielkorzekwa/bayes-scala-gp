Assumptions, these simplify the model equations
* All tasks are fully observed
* Z_i=Z_j for all P and Q variables
* covFunc is the same across all v and u processes]
* Z = X
* cogpPredict works for a single g function only

Questions:
1) How to initialise model parameters for svi: beta, w
2) Support the model for any number of outputs, currently it works for two outputs only
3) What should be the order of parameters learning?