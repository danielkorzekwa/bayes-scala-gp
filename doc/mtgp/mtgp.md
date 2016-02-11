# Multi Task Gaussian Process Regression

 It is almost like standard Gaussian Process Regression, but with hyper parameters shared between task Gaussian Processes. 

Notes for learning phase:
  * Log likelihood is approximated by loo-cv based on cavity distributions. See 5.5.3 in Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning, 2006
  * [Log likelihood gradient derivation](https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/doc/gpc/gpc-loglik.pdf)

* [Parameters learning ] (https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/src/test/scala/dk/gp/mtgp/learnMtGgHyperParamsTest.scala)