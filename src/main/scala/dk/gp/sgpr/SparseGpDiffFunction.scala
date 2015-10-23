package dk.gp.sgpr

import breeze.linalg.DenseMatrix
import dk.gp.cov.CovSEiso
import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import dk.gp.cov.utils.covDiagD
import dk.gp.cov.CovFunc
import dk.gp.sgpr.lb.calcLoglikWithD

case class SparseGpDiffFunction(x: DenseMatrix[Double], y: DenseVector[Double], u: DenseMatrix[Double], covFunc: CovFunc) extends DiffFunction[DenseVector[Double]] {

  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {

    val covFuncParams = params(0 until params.size - 1)
    val logNoiseStdDev = params(params.size - 1)
    
    val (loglik, loglikDKernel, loglikDLikNoise) = try {
      calcLoglikWithD(x, y, u, covFunc, covFuncParams, logNoiseStdDev)
    } catch {
      case e: Exception => (Double.NaN, covFuncParams.map(x => Double.NaN).toArray, Double.NaN)
    }

    val negativeD = DenseVector(loglikDKernel.map(x => -x) :+ (-loglikDLikNoise))
    (-loglik, negativeD)
  }

}