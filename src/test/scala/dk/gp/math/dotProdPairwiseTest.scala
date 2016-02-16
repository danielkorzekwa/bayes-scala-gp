package dk.gp.math

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import dk.bayes.math.linear.isIdentical

class dotProdPairwiseTest {

  @Test def test = {

    val x1 = DenseMatrix((1.0, 2.0), (3.0, 4.0))
    val x2 = DenseMatrix((5.0, 6.0), (7.0, 8.0), (9.0, 10.0))

    val dotProdMat = dotProdPairwise(x1.t, x2.t)

    assertTrue(isIdentical(dotProdMat, DenseMatrix((17.0, 23.0, 29.0), (39.0, 53.0, 67.0)), 0.01))
  }
}