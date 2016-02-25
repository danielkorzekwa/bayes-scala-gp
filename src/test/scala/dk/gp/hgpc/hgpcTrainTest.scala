package dk.gp.hgpc

import breeze.numerics._
import breeze.linalg.DenseVector
import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import com.typesafe.scalalogging.slf4j.LazyLogging
import dk.bayes.math.linear.isIdentical

class hgpcTrainTest extends LazyLogging {

  val (x, y, u) = getHgpcTestData()

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(log(1), log(1), log(1)) //log sf, logEllx1, logEllx2
  val mean = 0d

  val xTest = DenseMatrix((1.0, 4.0, -4.0), (1.0, -2.5, -1.3), (2.0, 4.0, -4.0), (2.0, -2.5, -1.3), (3.0, 4.0, -4.0), (3.0, -2.5, -1.3), (99.0, 4.0, -4.0), (99.0, -2.5, -1.3))

  @Test def test = {

    val model = HgpcModel(x, y, u, covFunc, covFuncParams, mean)
    val trainedModel = hgpcTrain(model, maxIter = 3)
    logger.info(s"trained cov params=${trainedModel.covFuncParams}, trained mean=${trainedModel.mean}")

    assertTrue("actual:" + trainedModel.covFuncParams, isIdentical(DenseVector(1.1589097518504272, 0.05547575752863218, 0.35914404525312227), trainedModel.covFuncParams, 0.0001))
    assertEquals(-0.5261, trainedModel.mean, 0.0001)

    val predicted = hgpcPredict(xTest, trainedModel)
    assertTrue(
      isIdentical(DenseVector(0.4262832412031917, 0.8839892446263581, 0.42628628427193305, 0.883989841119802, 0.42628502561601134, 0.8839894716952773, 0.42628523013193, 0.8839896446532274),
        predicted, 0.0001))

  }

}