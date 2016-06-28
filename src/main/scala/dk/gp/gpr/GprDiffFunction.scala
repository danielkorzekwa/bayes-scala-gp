package dk.gp.gpr

import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import scala.math._
import breeze.linalg.NotConvergedException
import breeze.linalg.MatrixNotSymmetricException

case class GprDiffFunction(initialGpModel: GprModel) extends DiffFunction[DenseVector[Double]] {

  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {

    try {
      val covFuncParams = DenseVector(params.toArray.dropRight(1))
      val noiseLogStdDev = params.toArray.last
      val gpModel = GprModel(initialGpModel.x, initialGpModel.y, initialGpModel.covFunc, covFuncParams, noiseLogStdDev, initialGpModel.meanFunc)

      val meanX = gpModel.meanFunc(gpModel.x)
      val kXX = gpModel.calcKXX()
      val kXXInv = gpModel.calcKXXInv(kXX)

      val f = -gprLoglik(meanX, kXX, kXXInv, gpModel.y)

      //calculate partial derivatives
      val covFuncCovElemWiseD = gpModel.covFunc.covD(gpModel.x, gpModel.x, gpModel.covFuncParams)
      val noiseCovElemWiseD = 2 * exp(2 * noiseLogStdDev) * DenseMatrix.eye[Double](gpModel.x.rows)
      val allParamsCovElemWiseD = covFuncCovElemWiseD :+ noiseCovElemWiseD

      val covFuncParamsD = gprLoglikD(meanX, kXXInv, gpModel.y, allParamsCovElemWiseD).map(d => -d)

      (f, covFuncParamsD)
    } catch {
      case e: NotConvergedException       => (Double.NaN, DenseVector.zeros[Double](params.size) * Double.NaN)
      case e: MatrixNotSymmetricException => (Double.NaN, DenseVector.zeros[Double](params.size) * Double.NaN)
    }
  }

}