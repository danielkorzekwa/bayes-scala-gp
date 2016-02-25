package dk.gp.hgpc

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.gp.gpc.GpcModel
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian
import dk.bayes.math.gaussian.canonical.SparseCanonicalGaussian
import breeze.linalg.CSCMatrix
import breeze.linalg.SparseVector
import dk.bayes.math.gaussian.canonical.CanonicalGaussian

class hgpcPredictTest {

  val (x, y, u) = getHgpcTestData()

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(1.219809, 0.052334, 0.171199) //log sf, logEllx1, logEllx2
  val mean = -2.8499

  val xTest = DenseMatrix((1.0, 4.0, -4.0), (1.0, -2.5, -1.3), (2.0, 4.0, -4.0), (2.0, -2.5, -1.3), (3.0, 4.0, -4.0), (3.0, -2.5, -1.3), (99.0, 4.0, -4.0), (99.0, -2.5, -1.3))

  @Test def test: Unit = {

    val model = HgpcModel(x, y, u, covFunc, covFuncParams, mean)
    val predicted = hgpcPredict(xTest, model)
    
    assertEquals(0.20623, predicted(0), 0.0001)
    assertEquals(0.8645, predicted(1), 0.0001)

    assertEquals(0.20623428, predicted(2), 0.0001)
    assertEquals( 0.864512843, predicted(3), 0.0001)

    assertEquals(0.2062340, predicted(4), 0.0001)
    assertEquals(0.86451368, predicted(5), 0.0001)

    assertEquals(0.2062, predicted(6), 0.0001)
    assertEquals(0.8645, predicted(7), 0.0001)

  }
}