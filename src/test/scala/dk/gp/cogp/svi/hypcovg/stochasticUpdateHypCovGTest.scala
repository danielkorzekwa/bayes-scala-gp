package dk.gp.cogp.svi.hypcovg

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel
import dk.gp.cogp.calcLBLoglik
import dk.gp.cogp.testutils.createCogpModel

class stochasticUpdateHypCovGTest {

  @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val (newHypParams, newHypParamsDelta) = stochasticUpdateHypCovG(j = 0, model, x, y)

    val newG = model.g.head.copy(covFuncParams = newHypParams, covFuncParamsDelta = newHypParamsDelta)
    val newModel = model.copy(g = Array(newG))

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-209.55036, loglik, 0.00001)
  }

  @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)

    val (newHypParams, newHypParamsDelta) = stochasticUpdateHypCovG(j = 0, model, x, y)

    val newG = model.g.head.copy(covFuncParams = newHypParams, covFuncParamsDelta = newHypParamsDelta)
    val newModel = model.copy(g = Array(newG))

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-6.710887560647833e10, loglik, 0.001)
  }

}