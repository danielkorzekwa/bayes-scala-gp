package dk.gp.math

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.cholesky

class logdetcholTest {

  @Test def test = {

    val m = DenseMatrix((1.0, 0.5, 0.7), (0.5, 2.0, 0.9), (0.7, 0.9, 3.0))

    assertEquals(1.408544, logdetchol(cholesky(m)), 0.00001)
  }
}