package dk.gp.cogp

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.MultivariateGaussian
import dk.gp.cov.utils.covDiag

case class CogpModel(g: Array[CogpGPVar], h: Array[CogpGPVar],
    beta: DenseVector[Double],betaDelta:DenseVector[Double],
    w: DenseMatrix[Double],wDelta:DenseMatrix[Double]) {

}