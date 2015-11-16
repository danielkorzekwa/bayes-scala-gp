package dk.gp.cov

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics._
import breeze.linalg._
import scala.util.Random

class CovSEisoTest {

  private val covFunc = new CovSEiso()
  val covFuncParams = DenseVector(log(2.0), log(10.0))

  /**
   * F
   * Tests for cov()
   */

  @Test def test_1D_cov = {

    assertEquals(4, covFunc.cov(DenseMatrix(3d), DenseMatrix(3d), covFuncParams)(0, 0), 0.0001)
    assertEquals(3.8239, covFunc.cov(DenseMatrix(2d), DenseMatrix(5d), covFuncParams)(0, 0), 0.0001)
    assertEquals(2.9045, covFunc.cov(DenseMatrix(2d), DenseMatrix(10d), covFuncParams)(0, 0), 0.0001)
    assertEquals(1.3181, new CovSEiso().cov(DenseMatrix(2d), DenseMatrix(300d), covFuncParams = DenseVector(log(2.0), log(200d)))(0, 0), 0.0001)
  }

  @Test def multi_dim_cov = {

    val x1 = DenseVector.fill(4)(1d).toDenseMatrix
    val x2 = DenseVector.fill(4)(2d).toDenseMatrix

    val covValue = covFunc.cov(x1, x2, covFuncParams)(0, 0)
    assertEquals(3.92079, covValue, 0.00001)
  }

  @Test def multi_dim_cov_2 = {
    val rand = new Random(4656)
    val n = 2000
    val x1 = DenseVector.fill(n)(rand.nextDouble).toDenseMatrix
    val x2 = DenseVector.fill(n)(rand.nextDouble).toDenseMatrix

    val covValue = covFunc.cov(x1, x2, covFuncParams)(0, 0)
    assertEquals(0.71287, covValue, 0.00001)
  }

  @Test def perf_test_1d_cov = {

    val x1 = DenseMatrix(10d)
    val x2 = DenseMatrix(2d)
    (1L to 200L * 50 * 50).foreach(_ => covFunc.cov(x1, x2, covFuncParams))

  }

  /**
   * Tests for df_dSf
   *
   */
  @Test def test_1D_df_dSf = {

    assertEquals(8, covFunc.covD(DenseMatrix(3d), DenseMatrix(3d), covFuncParams)(0)(0, 0), 0.0001)
    assertEquals(7.6479, covFunc.covD(DenseMatrix(2d), DenseMatrix(5d), covFuncParams)(0)(0, 0), 0.0001)
    assertEquals(5.8091, covFunc.covD(DenseMatrix(2d), DenseMatrix(10d), covFuncParams)(0)(0, 0), 0.0001)
    assertEquals(2.6363, new CovSEiso().covD(DenseMatrix(2d), DenseMatrix(300d), DenseVector(log(2), log(200)))(0)(0, 0), 0.0001)
  }

  @Test def multi_dim_df_dSf = {
    val rand = new Random(4656)
    val n = 2000
    val x1 = DenseVector.fill(n)(rand.nextDouble).toDenseMatrix
    val x2 = DenseVector.fill(n)(rand.nextDouble).toDenseMatrix

    val covValue = covFunc.covD(x1, x2, covFuncParams)(0)(0, 0)
    assertEquals(1.4257, covValue, 0.0001)
  }

  @Test def perf_test_1d_df_dSf = {

    val x1 = DenseMatrix(10d)
    val x2 = DenseMatrix(2d)
    (1L to 200L * 50 * 50).foreach(_ => covFunc.covD(x1, x2, covFuncParams))

  }

  /**
   * Tests for df_dEll
   */
  @Test def test_1D_df_dEll = {

    assertEquals(0, covFunc.covD(DenseMatrix(3d), DenseMatrix(3d), covFuncParams)(1)(0, 0), 0.0001)
    assertEquals(0.3441, covFunc.covD(DenseMatrix(2d), DenseMatrix(5d), covFuncParams)(1)(0, 0), 0.0001)
    assertEquals(1.8589, covFunc.covD(DenseMatrix(2d), DenseMatrix(10d), covFuncParams)(1)(0, 0), 0.0001)
    assertEquals(2.9264, new CovSEiso().covD(DenseMatrix(2d), DenseMatrix(300d), DenseVector(log(2d), log(200d)))(1)(0, 0), 0.0001)
  }

  @Test def multi_dim_df_dEll = {
    val rand = new Random(4656)
    val n = 2000
    val x1 = DenseVector.fill(n)(rand.nextDouble).toDenseMatrix
    val x2 = DenseVector.fill(n)(rand.nextDouble).toDenseMatrix

    val covValue = covFunc.covD(x1, x2, covFuncParams)(1)(0, 0)
    assertEquals(2.4590, covValue, 0.0001)
  }

  @Test def perf_test_1d_df_dEll = {

    val x1 = DenseMatrix(10d)
    val x2 = DenseMatrix(2d)
    (1L to 200L * 50 * 50).foreach(_ => covFunc.covD(x1, x2, covFuncParams))

  }

  @Test def perf_test_1d_df_dEll_matrix_input = {

    val x = DenseMatrix.zeros[Double](1000, 1)
    (1 to 10).foreach(_ => covFunc.covD(x, x, covFuncParams))
  }

  /**
   * Perf test for cov(Double,Double)
   */

  @Test def perf_cov_Double_Double = {
    val x1 = DenseMatrix(10.0)
    val x2 = DenseMatrix(200.0)

    for (i <- 1 to 40000) covFunc.cov(x1, x2, covFuncParams)
  }

}