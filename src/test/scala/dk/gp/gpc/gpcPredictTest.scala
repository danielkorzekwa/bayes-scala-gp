package dk.gp.gpc

import java.io.File

import org.junit.Assert.assertEquals
import org.junit.Test

import breeze.linalg.DenseVector
import breeze.linalg.csvread

/**
 * Classification example from http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */
class gpcPredictTest {

  val x = csvread(new File("src/test/resources/gpml/classification_x.csv"))
  val y = csvread(new File("src/test/resources/gpml/classification_y.csv")).toDenseVector
  val t = csvread(new File("src/test/resources/gpml/classification_test.csv"))

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(1.219809, 0.052334, 0.171199) //log sf, logEllx1, logEllx2
  val mean = -2.8499

  @Test def test = {

    val model = GpcModel(x, y, covFunc, covFuncParams, mean)
    val predicted = gpcPredict(t, model)
    assertEquals(0.20625, predicted(6480), 0.0001) // t = [4 -4]
    assertEquals(0.18870, predicted(1255), 0.0001) //t = [-2.5 0]
    assertEquals(0.87060, predicted(1242), 0.0001) //t = [-2.5 -1.3]
  }
}