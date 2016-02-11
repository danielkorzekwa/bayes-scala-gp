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

  val x = csvread(new File("src/test/resources/gpml/classification_x.csv"))//(0 to 1, ::)
  val y = csvread(new File("src/test/resources/gpml/classification_y.csv")).toDenseVector//(0 to 1)
  val t = csvread(new File("src/test/resources/gpml/classification_test.csv"))

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(log(1), log(1), log(1)) //log sf, logEllx1, logEllx2
  val mean = 0

  @Test def test = {

    val model = GpcModel(x, y, covFunc, covFuncParams, mean)
    val trainedModel = gpcTrain(model, maxIter = 10)

    assertEquals(-1.73052, trainedModel.gpMean, 0.0001)

    assertEquals(4.28354, trainedModel.covFuncParams(0), 0.0001) //logSf
    assertEquals(0.55933, trainedModel.covFuncParams(1), 0.0001) //logEllx1
    assertEquals(1.74289, trainedModel.covFuncParams(2), 0.0001) //logEllx2

    val predicted = gpcPredict(t, trainedModel)
    
    assertEquals(0.18973, predicted(6480), 0.0001) // t = [4 -4]
    assertEquals(0.3966, predicted(1255), 0.0001) //t = [-2.5 0]
    assertEquals(0.91849, predicted(1242), 0.0001) //t = [-2.5 -1.3]
  }
}