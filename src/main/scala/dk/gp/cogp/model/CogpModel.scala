package dk.gp.cogp.model

import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.math.MultivariateGaussian
import dk.gp.cov.utils.covDiag
import breeze.linalg.inv
import breeze.stats._
import breeze.linalg._
import dk.gp.math.invchol
import scala.util.Random

case class CogpModel(g: Array[CogpGPVar], h: Array[CogpGPVar],
                     beta: DenseVector[Double], betaDelta: DenseVector[Double],
                     w: DenseMatrix[Double], wDelta: DenseMatrix[Double]) {

  if (g.size > 0) {
    require(w.rows == h.size, "w.rows must equal to h.size")
    require(w.cols == g.size, "w.cols must equal to g.size")
  } else require(w.size == 0, "w.rows must be empty if g.size==0")

  require(beta.findAll(_.isNaN).size == 0, "Beta parameter is NaN:" + beta)
  require(beta.findAll(_ <= 0).size == 0, "Beta parameter must be positive:" + beta)

}

object CogpModel {

  def apply(gVariables: Array[CogpGPVar], hVariables: Array[CogpGPVar],
            beta: DenseVector[Double], w: DenseMatrix[Double]): CogpModel = {

    val betaDelta = DenseVector.zeros[Double](beta.size)
    val wDelta = DenseMatrix.zeros[Double](w.rows, w.cols)

    CogpModel(gVariables, hVariables, beta, betaDelta, w, wDelta)
  }

}