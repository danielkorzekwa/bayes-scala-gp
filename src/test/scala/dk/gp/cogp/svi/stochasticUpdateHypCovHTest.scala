package dk.gp.cogp.svi

import org.junit._
import org.junit.Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cogp.testutils.createCogpToyModel
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.testutils._

class stochasticUpdateHypCovHTest {

   @Test def test_5_data_points = {

    val data = loadToyModelData(n = 5)

    val model = createCogpToyModel(data)
    
      val (newHypParams0, newHypParamsDelta0) = stochasticUpdateHypCovH(i = 0, LowerBound(model,data))
    val newH0 = model.h(0).copy(covFuncParams = newHypParams0, covFuncParamsDelta = newHypParamsDelta0)

    val (newHypParams1, newHypParamsDelta1) = stochasticUpdateHypCovH(i = 1, LowerBound(model,data))
    val newH1 = model.h(1).copy(covFuncParams = newHypParams1, covFuncParamsDelta = newHypParamsDelta1)

    val newModel = model.copy(h = Array(newH0, newH1))

    val loglik = calcLBLoglik(LowerBound(newModel,data))
    assertEquals(-121201.056426, loglik, 0.00001)
  }
  
    @Test def test_40_data_points = {

    val data = loadToyModelData(n = 40)

    val model = createCogpToyModel(data)
    
      val (newHypParams0, newHypParamsDelta0) = stochasticUpdateHypCovH(i = 0, LowerBound(model,data))
    val newH0 = model.h(0).copy(covFuncParams = newHypParams0, covFuncParamsDelta = newHypParamsDelta0)

    val (newHypParams1, newHypParamsDelta1) = stochasticUpdateHypCovH(i = 1,  LowerBound(model,data))
    val newH1 = model.h(1).copy(covFuncParams = newHypParams1, covFuncParamsDelta = newHypParamsDelta1)

    val newModel = model.copy(h = Array(newH0, newH1))

    val loglik = calcLBLoglik(LowerBound(newModel,data))
    assertEquals(-9.210001184817e7, loglik, 0.00001)
  }
  

}