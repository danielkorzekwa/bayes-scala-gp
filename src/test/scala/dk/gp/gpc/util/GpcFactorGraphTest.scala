package dk.gp.gpc.util

import org.junit._
import Assert._
import breeze.linalg._
import java.io._
import dk.gp.gpc.TestCovFunc
import dk.gp.gpc.GpcModel
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian

class GpcFactorGraphTest {

  val x = csvread(new File("src/test/resources/gpml/classification_x.csv"))
  val y = csvread(new File("src/test/resources/gpml/classification_y.csv")).toDenseVector

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(1.219809, 0.052334, 0.171199) //log sf, logEllx1, logEllx2
  val gpMean = -2.8499

  val gpcModel = GpcModel(x, y, covFunc, covFuncParams, gpMean)

  @Test def test = {

    val gpcFactorGraph = GpcFactorGraph(gpcModel)
    val (calib, iter) = calibrateGpcFactorGraph(gpcFactorGraph, maxIter = 100)

    assertTrue(calib)
    assertEquals(44, iter)

    val fVariable = gpcFactorGraph.fVariable.get.asInstanceOf[DenseCanonicalGaussian]

    assertEquals(-1.411514, fVariable.mean(0), 0.00001)
    assertEquals(-1.28964, fVariable.mean(1), 0.00001)
    assertEquals(-6.3971, fVariable.mean(2), 0.0001)
  }
}