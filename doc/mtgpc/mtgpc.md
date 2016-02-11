# Multi Task Gaussian Process Classification

 It is almost like standard Gaussian Process Classification, but with hyper parameters shared between task Gaussian Processes. 

Notes for learning phase:
  * Log likelihood is approximated by loo-cv based on cavity distributions. See 5.5.3 in Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning, 2006
  * [Log likelihood gradient derivation](https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/doc/gpc/gpc-loglik.pdf)

* [Train multi task gpc model] (https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/src/test/scala/dk/gp/mtgpc/mtgpcTrainTest.scala)
* [Multi task gpc model prediction] (https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/src/test/scala/dk/gp/mtgpc/mtgpcPredictTest.scala)