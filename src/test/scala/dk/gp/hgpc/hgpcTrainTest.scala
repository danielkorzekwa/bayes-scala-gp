package dk.gp.hgpc

import breeze.numerics._
import breeze.linalg.DenseVector
import org.junit._
import Assert._
import breeze.linalg.DenseMatrix

class hgpcTrainTest {

  val (x, y, u) = getHgpcTestData()

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(log(1), log(1), log(1)) //log sf, logEllx1, logEllx2
  val mean = 0d

  val xTest = DenseMatrix((1.0, 4.0, -4.0), (1.0, -2.5, -1.3), (2.0, 4.0, -4.0), (2.0, -2.5, -1.3), (3.0, 4.0, -4.0), (3.0, -2.5, -1.3), (99.0, 4.0, -4.0), (99.0, -2.5, -1.3))

  @Ignore @Test def test = {

    val model = HgpcModel(x, y, u, covFunc, covFuncParams, mean)
    val trainedModel = hgpcTrain(model)
    val predicted = hgpcPredict(xTest, trainedModel)

    println(predicted)

  }

}