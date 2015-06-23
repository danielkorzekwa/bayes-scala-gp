package dk.gp.cogp

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg._
import breeze.numerics._
import dk.gp.cov.CovSEiso

class predictTest {

  val x: DenseMatrix[Double] = DenseVector.rangeD(-10, 10, 1).toDenseMatrix.t
  val y = DenseVector.horzcat(DenseVector.zeros[Double](x.size) + 1d, DenseVector.zeros[Double](x.size) + 2d)

  val covFuncParams = DenseVector(log(1), log(1))
  val covFunc = CovSEiso()

  @Test def test = {

    val model = cogp(x, y, covFunc, covFuncParams)
    val predictedY = predict(x, model)

    println(predictedY)
  }
}