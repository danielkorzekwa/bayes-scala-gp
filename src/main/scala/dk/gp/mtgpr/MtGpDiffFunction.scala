package dk.gp.mtgpr

import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import breeze.linalg.DenseMatrix
import dk.gp.cov.CovFunc
import dk.gp.gpr.GprDiffFunction
import dk.gp.gpr.GprModel
import dk.gp.gpr.GprDiffFunction
import breeze.linalg.sum

case class MtGpDiffFunction(gpDiffFunctions: Seq[GprDiffFunction]) extends DiffFunction[DenseVector[Double]] {

  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {

    val loglikWithD = gpDiffFunctions.map(gpDiffFunc => gpDiffFunc.calculate(params))

    val totalLoglik = loglikWithD.map(_._1).sum
    val totalGrad = sum(loglikWithD.map(_._2))

    (totalLoglik, totalGrad)

  }

}

object MtGpDiffFunction {
  def apply(model:MtGprModel): MtGpDiffFunction = {
    
    val gpDiffFunctions = model.data.map{taskData =>
      val x = taskData(::,0 until taskData.cols-1)
      val y = taskData(::,taskData.cols-1)
      val gprModel = GprModel(x, y, model.covFunc, model.covFuncParams, model.likNoiseLogStdDev)
      val gpDiffFunction = GprDiffFunction(gprModel)
      gpDiffFunction
    }
   
    MtGpDiffFunction(gpDiffFunctions)
  }

}