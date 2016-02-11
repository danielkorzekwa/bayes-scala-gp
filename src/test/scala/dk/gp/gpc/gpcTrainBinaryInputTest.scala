package dk.gp.gpc

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovSEiso
import breeze.numerics._

class gpcTrainBinaryInputTest {

  val data = csvread(new File("src/test/resources/gpc/gpc_binary_input.csv"), skipLines = 1)
  val x: DenseMatrix[Double] = data(::, 0 to 0)
  val y = data(::, 1)

  val covFunc = CovSEiso()
  val covFuncParams = DenseVector(log(0.1), log(0.3)) //log sf, logEll
  val mean = 0

  @Test def test = {

    val model = GpcModel(x, y, covFunc, covFuncParams, mean)

    val trainedModel = gpcTrain(model, maxIter = 10)
    
    println("prior params=" + model.covFuncParams)
    println("learned params=" + trainedModel.covFuncParams)
    println("learned gp mean=" + trainedModel.gpMean)

  }
}