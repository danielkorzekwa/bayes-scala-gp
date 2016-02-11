package dk.gp.gpc.util

import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import breeze.optimize.ApproximateGradientFunction
import util._
import breeze.linalg._
import dk.gp.gpc.GpcModel

case class GpcLowerboundDiffFunction(initialGpcModel: GpcModel) extends DiffFunction[DenseVector[Double]] {

  def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
    try {
      val currCovFuncParams = DenseVector(x.toArray.dropRight(1))
      val currMean = x.toArray.last
      val currModel = initialGpcModel.copy(covFuncParams = currCovFuncParams, gpMean = currMean)

      val gpcFactorGraph = GpcFactorGraph(currModel)
      val (calib, iters) = calibrateGpcFactorGraph(gpcFactorGraph, maxIter = 10)
      val loglik = -calcGPCLoglik(gpcFactorGraph)

      val loglikCovD = -calcGPCLoglikD(gpcFactorGraph)
      val loglikMeanD = -calcGPCLoglikMeanD(gpcFactorGraph)

      val grad: DenseVector[Double] = DenseVector(loglikCovD.toArray :+ loglikMeanD)

      (loglik, grad)

    } catch {
      case e: NotConvergedException    => (Double.NaN, DenseVector.zeros[Double](x.size) * Double.NaN)
      case e: IllegalArgumentException => e.printStackTrace(); System.exit(-1); (Double.NaN, DenseVector.zeros[Double](x.size) * Double.NaN)
    }
  }

}