package dk.gp.math

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix

case class MultivariateGaussian(m:DenseVector[Double],v:DenseMatrix[Double])