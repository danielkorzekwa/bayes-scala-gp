package dk.gp.mtgpr

import java.io.File

import org.junit._
import org.junit.Assert._

import breeze.linalg._
import breeze.numerics._

class mtgprTrainTest {

  //[x,y]
  val data = csvread(new File("src/test/resources/gp/gpml_regression_data.csv"), skipLines = 1)
  val x = data(::, 0 to 0)
  val y = data(::, 1)

  val x1 = DenseMatrix.horzcat(DenseMatrix.zeros[Double](y.size, 1) + 1.0, x)
  val x2 = DenseMatrix.horzcat(DenseMatrix.zeros[Double](y.size, 1) + 2.0, x)

  val allX = DenseMatrix.vertcat(x1, x2)
  val allY = DenseVector.vertcat(y, y)

  val covFunc = TestMtGprCovFunc()
  val covFuncParams = DenseVector(log(1d), log(1))
  val likNoiseLogStdDev = log(0.1)

  val mtGprModel = MtGprModel(allX, allY, covFunc, covFuncParams, likNoiseLogStdDev)

  @Test def test = {

    val trainedModel = mtgprTrain(mtGprModel)

    assertEquals(0.68594, trainedModel.covFuncParams(0), 0.0001)
    assertEquals(-0.99340, trainedModel.covFuncParams(1), 0.0001)
    assertEquals(-1.9025, trainedModel.likNoiseLogStdDev, 0.0001)
  }
}