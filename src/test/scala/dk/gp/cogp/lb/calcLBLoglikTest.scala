package dk.gp.cogp.lb

import org.junit.Assert.assertEquals
import org.junit.Test

import breeze.linalg.DenseVector
import breeze.linalg.csvread
import breeze.numerics.log
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.testutils.loadToyModelData
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso

class calcLBLoglikTest {


  val covFuncG: Array[CovFunc] = Array(CovSEiso())
  val cofFuncGParams = Array(DenseVector(log(1), log(1)))

  val covFuncH: Array[CovFunc] = Array(CovSEiso(), CovSEiso())
  val covFuncHParams = Array(DenseVector(log(1), log(1e-10)), DenseVector(log(1), log(1e-10)))

  @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)

    val loglik = calcLBLoglik(LowerBound(model,data))
    assertEquals(-121201.05673, loglik, 0.00001)
  }

  @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)

    val loglik = calcLBLoglik(LowerBound(model,data))
    assertEquals(-9.210001189441e7, loglik, 0.0001)
  }

}