package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.MultivariateGaussian

case class LBState(u:Array[MultivariateGaussian],v:Array[MultivariateGaussian],
    beta: DenseVector[Double],betaDelta:DenseVector[Double],
    w: DenseMatrix[Double],wDelta:DenseMatrix[Double])