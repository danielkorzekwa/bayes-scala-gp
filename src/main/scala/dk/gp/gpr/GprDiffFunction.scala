package dk.gp.gpr

import breeze.linalg.DenseVector
import breeze.optimize.DiffFunction
import dk.gp.cov.CovFunc
import breeze.linalg.DenseMatrix
import scala.math._

case class GprDiffFunction(initialGpModel: GprModel) extends DiffFunction[DenseVector[Double]] {
  
  def calculate(params: DenseVector[Double]): (Double, DenseVector[Double]) = {

    val covFuncParams = DenseVector(params.toArray.dropRight(1))
    val noiseLogStdDev = params.toArray.last
    val gpModel = GprModel(initialGpModel.x, initialGpModel.y, initialGpModel.covFunc, covFuncParams, noiseLogStdDev, initialGpModel.mean)
    
    val f = -gprLoglik(gpModel.meanX, gpModel.kXX, gpModel.kXXInv, gpModel.y)

    //calculate partial derivatives
    val covFuncCovElemWiseD = gpModel.covFunc.covD(gpModel.x,gpModel.x,gpModel.covFuncParams)
    val noiseCovElemWiseD = 2 * exp(2 * noiseLogStdDev) * DenseMatrix.eye[Double](gpModel.x.rows)
    val allParamsCovElemWiseD = covFuncCovElemWiseD :+ noiseCovElemWiseD

    val covFuncParamsD = gprLoglikD(gpModel.meanX, gpModel.kXXInv, gpModel.y, allParamsCovElemWiseD).map(d => -d)

    (f, covFuncParamsD)
  }

}