package dk.gp.mtgpc

import org.junit.Assert.assertEquals
import org.junit.Test

import breeze.linalg.DenseVector
import breeze.numerics.log

class learnMtGgcHyperParamsTest {

  val (x, y) = getMtGpcTestData()
println(x.rows)
  val covFunc = TestMtGpcCovFunc()
  val covFuncParams = DenseVector(log(1), log(1), log(1)) //log sf, logEllx1, logEllx2
  val gpMean = 0

  @Test def test = {

    val (learnedCovParams, learnedGpMean) = learnMtGgcHyperParams(x, y, covFunc, covFuncParams, gpMean)
    assertEquals(-6.5124, learnedGpMean, 0.0001)

    assertEquals(3.9167, learnedCovParams(0), 0.0001) //logSf
    assertEquals(1.3032, learnedCovParams(1), 0.0001) //logEllx1
    assertEquals(1.1069, learnedCovParams(2), 0.0001) //logEllx2
  }
}