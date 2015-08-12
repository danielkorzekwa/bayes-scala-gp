package dk.gp.cogp

import dk.gp.math.MultivariateGaussian
import dk.gp.cov.CovFunc
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix

case class CogpGPVar(z: DenseMatrix[Double], u: MultivariateGaussian, covFunc: CovFunc, covFuncParams: DenseVector[Double],covFuncParamsDelta: DenseVector[Double])