package dk.gp.gpc

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test
import breeze.linalg.DenseVector
import breeze.linalg.csvread
import breeze.numerics.log
import org.junit.Ignore

/**
 * Classification example from http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */
class gpcTrainTest {

  val x = csvread(new File("src/test/resources/gpml/classification_x.csv"))
  val y = csvread(new File("src/test/resources/gpml/classification_y.csv")).toDenseVector
  val t = csvread(new File("src/test/resources/gpml/classification_test.csv"))

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(log(1), log(1), log(1)) //log sf, logEllx1, logEllx2
  val mean = 0

  //@TODO failing on CI
  @Ignore @Test def test = {

    val model = GpcModel(x, y, covFunc, covFuncParams, mean)
    val trainedModel = gpcTrain(model, maxIter = 10)

    assertEquals(-1.5206, trainedModel.mean, 0.0001)

    assertEquals(3.6588, trainedModel.covFuncParams(0), 0.0001) //logSf
    assertEquals(0.5046, trainedModel.covFuncParams(1), 0.0001) //logEllx1
    assertEquals(1.35238, trainedModel.covFuncParams(2), 0.0001) //logEllx2

    val predicted = gpcPredict(t, trainedModel)

    assertEquals(0.2595, predicted(6480), 0.0001) // t = [4 -4]
    assertEquals(0.32380, predicted(1255), 0.0001) //t = [-2.5 0]
    assertEquals(0.91336, predicted(1242), 0.0001) //t = [-2.5 -1.3]
  }
}