package dk.gp.util

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import dk.bayes.math.linear.isIdentical

class calcUniqueRowsMatrixTest {

  @Test def test = {

    val m = DenseMatrix((0.2, 0.5), (0.21, 0.5), (0.2, 0.5))
    val uniqueRowsM = calcUniqueRowsMatrix(m)

    assertTrue(isIdentical(DenseMatrix((0.2, 0.5), (0.21, 0.5)), uniqueRowsM, 0.000001))

  }
}