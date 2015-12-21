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
    assertTrue("actual:" + trainedModel.covFuncParams, isIdentical(DenseVector(1.1828702973619143, 0.06499356917713628, 0.37643910766297717), trainedModel.covFuncParams, 0.0001))
    assertEquals(-0.5336, trainedModel.mean, 0.0001)

    val predicted = hgpcPredict(xTest, trainedModel)
    assertTrue(isIdentical(DenseVector(0.47687091736985365, 0.884525238578147, 0.47687438858256437, 0.884525822729599, 0.47687293598303815, 0.8845254746661855, 0.42611968744238105, 0.8843541153804271), predicted, 0.0001))

  }

}