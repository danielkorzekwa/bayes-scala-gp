package dk.gp.mtgpc

import breeze.optimize.DiffFunction
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import dk.gp.cov.CovFunc
import dk.gp.gpc.GpcDiffFunction
import dk.gp.gpc.GpcModel
import breeze.linalg._

case class MtGpcDiffFunction(x: DenseMatrix[Double], y: DenseVector[Double], covFunc: CovFunc, initialCovFuncParams: DenseVector[Double], initialGpMean: Double) extends DiffFunction[DenseVector[Double]] {

  private val gpDiffFunctions = createGpDiffFunctions()

  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {
      val loglikWithD = gpDiffFunctions.map(gpDiffFunc => gpDiffFunc.calculate(params))

    val totalLoglik = loglikWithD.map(_._1).sum
    val totalGrad = sum(loglikWithD.map(_._2))

    (totalLoglik, totalGrad)
  }

  private def createGpDiffFunctions(): Seq[GpcDiffFunction] = {

    val taskIds = x(::, 0).toArray.distinct

    val gpDiffFunctions = taskIds.map { cId =>
      val idx = x(::, 0).findAll { x => x == cId }
      val taskX = x(idx, ::).toDenseMatrix
      val taskY = y(idx).toDenseVector

      val model = GpcModel(taskX, taskY, covFunc, initialCovFuncParams, initialGpMean)
      val gpDiffFunction = GpcDiffFunction(model)
      gpDiffFunction
    }
    gpDiffFunctions
  }
}