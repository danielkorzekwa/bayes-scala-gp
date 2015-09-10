package dk.gp.cogp.lb

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Test
import breeze.linalg.DenseVector
import breeze.linalg.csvread
import breeze.numerics.log
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik

class calcLBLoglikTest {

  val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
  val x = data(::, 0).toDenseMatrix.t
  val y = data(::, 1 to 2)

  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)

    val loglik = calcLBLoglik(LowerBound(model,x), y)
    assertEquals(-121201.05673, loglik, 0.00001)
  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpToyModel(x, y)

    val loglik = calcLBLoglik(LowerBound(model,x), y)
    assertEquals(-9.210001189441e7, loglik, 0.0001)
  }

}