package dk.gp.hgpc.factorgraph2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import breeze.linalg.DenseVector
import dk.gp.hgpc.HgpcModel
import dk.gp.hgpc.TestCovFunc
import dk.gp.hgpc.getHgpcTestData
import dk.gp.hgpc.util.HgpcFactorGraph2
import dk.gp.hgpc.util.calibrateHgpcFactorGraph2
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian

class HgpcFactorGraph2Test {

  val (x, y, u) = getHgpcTestData()

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(1.219809, 0.052334, 0.171199) //log sf, logEllx1, logEllx2
  val mean = -2.8499

  val hgpcModel = HgpcModel(x, y, u, covFunc, covFuncParams, mean)

  @Test def test: Unit = {

    val hgpcFactorGraph = HgpcFactorGraph2(hgpcModel)
    val (calib, iter) = calibrateHgpcFactorGraph2(hgpcFactorGraph,maxIter=100)

    assertTrue(calib)
    assertEquals(6, iter)

    assertEquals(-1.70451, hgpcFactorGraph.uVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(0), 0.00001)
    assertEquals(-1.26741, hgpcFactorGraph.uVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(1), 0.00001)
    assertEquals(-6.40155, hgpcFactorGraph.uVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(2), 0.00001)
  }
}