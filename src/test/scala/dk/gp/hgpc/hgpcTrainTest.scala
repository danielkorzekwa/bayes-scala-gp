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
    assertTrue("actual:" + trainedModel.covFuncParams, isIdentical(DenseVector(1.17334, 0.0666, 0.378324), trainedModel.covFuncParams, 0.0001))
    assertEquals(-0.5329, trainedModel.mean, 0.0001)

    val predicted = hgpcPredict(xTest, trainedModel)
    
    assertTrue(isIdentical(DenseVector(0.47663773073693394, 0.8811799001665213, 0.4740205824496625, 0.9412584617678159, 0.4765424710571, 0.8805323926428879, 0.4256003932592067, 0.8836453473885897),predicted,0.0001))

  }

}