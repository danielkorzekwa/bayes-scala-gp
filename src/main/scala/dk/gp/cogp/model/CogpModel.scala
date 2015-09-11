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
                     w: DenseMatrix[Double], wDelta: DenseMatrix[Double]) 
                     
object CogpModel {

  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], z: DenseMatrix[Double], gVariables: Array[CogpGPVar], hVariables: Array[CogpGPVar],
            beta: DenseVector[Double], w: DenseMatrix[Double]): CogpModel = {

    val betaDelta = DenseVector.zeros[Double](beta.size)
    val wDelta = DenseMatrix.zeros[Double](w.rows, w.cols)

    CogpModel(gVariables, hVariables, beta, betaDelta, w, wDelta)
  }

}