package dk.gp.mtgp

import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import breeze.linalg.DenseMatrix
import dk.gp.cov.CovFunc
import dk.gp.gpr.GprDiffFunction
import dk.gp.gpr.GprModel
import dk.gp.gpr.GprDiffFunction
import breeze.linalg.sum

case class MtGpDiffFunction(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, initialCovFuncParams: DenseVector[Double], initialLikNoiseLogStdDev: Double) extends DiffFunction[DenseVector[Double]] {

  private val gpDiffFunctions = createGpDiffFunctions()

  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {

    val loglikWithD = gpDiffFunctions.map(gpDiffFunc => gpDiffFunc.calculate(params))

    val totalLoglik = loglikWithD.map(_._1).sum
    val totalGrad = sum(loglikWithD.map(_._2))

    (totalLoglik, totalGrad)

  }

  private def createGpDiffFunctions(): Seq[GprDiffFunction] = {

    val taskIds = x(::, 0).toArray.distinct

    val gpDiffFunctions = taskIds.map { cId =>
      val idx = x(::, 0).findAll { x => x == cId }
      val taskX = x(idx, ::).toDenseMatrix
      val taskY = y(idx).toDenseVector

      val model = GprModel(taskX, taskY, covFunc, initialCovFuncParams, initialLikNoiseLogStdDev)
      val gpDiffFunction = GprDiffFunction(model)
      gpDiffFunction
    }
    gpDiffFunctions
  }
}