package dk.gp.math

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

class sqDistTest {

  @Test def test_david = {
    val x = DenseVector.rangeD(0, 4000, 1).toDenseMatrix

    val now = System.currentTimeMillis();
    for (i <- 1 to 10) sqDist(x, x)
    println("[1x4000]" + (System.currentTimeMillis() - now))
  }

  @Test def test_david2 = {
    val x = DenseMatrix.rand(5, 200)

    val now = System.currentTimeMillis();
    for (i <- 1 to 1000) sqDist(x, x)
    println("[5x200]" + (System.currentTimeMillis() - now))
  }

  @Test def test_xx_1D = {

    val x = DenseMatrix(1.0, 2.0, 3.0).t

    val expected = DenseMatrix((0.0, 1.0, 4.0), (1.0, 0.0, 1.0), (4.0, 1.0, 0.0))
    assertEquals(expected, sqDist(x, x))

  }

  @Test def test_xx_2D = {

    val x = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val expected = DenseMatrix((0.0, 2.0, 8.0), (2.0, 0.0, 2.0), (8.0, 2.0, 0.0))
    assertEquals(expected, sqDist(x, x))

  }

  @Test def test_x1_x2_1D = {

    val x1 = DenseMatrix(1.0, 2.0, 3.0).t
    val x2 = DenseMatrix(4.0, 5.0).t

    val expected = DenseMatrix((9.0, 16.0), (4.0, 9.0), (1.0, 4.0))
    assertEquals(expected, sqDist(x1, x2))

  }

  @Test def test_x1_x2_2D = {

    val x1 = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val x2 = DenseMatrix((7.0, 8.0), (9.0, 10.0))

    val expected = DenseMatrix((61.0, 85.0), (41.0, 61.0), (25.0, 41.0))
    assertEquals(expected, sqDist(x1, x2))

  }
}