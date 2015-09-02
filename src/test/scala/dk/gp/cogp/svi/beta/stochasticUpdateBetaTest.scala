package dk.gp.cogp.svi.beta

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovFunc
import dk.gp.cov.CovSEiso
import breeze.numerics.log
import dk.gp.cogp.CogpModel
import dk.gp.cogp.calcLBLoglik
import dk.gp.cogp.svi.w.stochasticUpdateW
import dk.gp.cogp.testutils.createCogpModel

class stochasticUpdateBetaTest {
  
   @Test def test_5_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)
    
     val (newBeta, newBetaDelta) = stochasticUpdateBeta(model, x, y)

    val newModel = model.copy(beta = newBeta)

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-121201.191556, loglik, 0.000001)
  }
   
     @Test def test_40_data_points = {

    val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 39, ::)
    val x = data(::, 0).toDenseMatrix.t
    val y = data(::, 1 to 2)

    val model = createCogpModel(x, y)
    
     val (newBeta, newBetaDelta) = stochasticUpdateBeta(model, x, y)

    val newModel = model.copy(beta = newBeta)

    val loglik = calcLBLoglik(newModel, x, y)
    assertEquals(-9.21000394392e7, loglik, 0.0001)
  }
  
  
}