package dk.gp.cogp.model

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.MultivariateGaussian

case class CogpModelParams(u:Array[MultivariateGaussian],v:Array[MultivariateGaussian],beta: DenseVector[Double],w: DenseMatrix[Double])