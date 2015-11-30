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
    assertTrue(isIdentical(DenseVector(1.1782774408943526, 0.0730341129940166, 0.37893991744472977), trainedModel.covFuncParams, 0.0001))
    assertEquals(-0.5343, trainedModel.mean, 0.0001)

    val predicted = hgpcPredict(xTest, trainedModel)

    assertEquals(0.47606, predicted(0), 0.0001)
    assertEquals(0.8812, predicted(1), 0.0001)

    assertEquals(0.47336, predicted(2), 0.0001)
    assertEquals(0.9415, predicted(3), 0.0001)

    assertEquals(0.4759, predicted(4), 0.0001)
    assertEquals(0.8808, predicted(5), 0.0001)

    assertEquals(0.42541, predicted(6), 0.0001)
    assertEquals(0.88394, predicted(7), 0.0001)

  }

}