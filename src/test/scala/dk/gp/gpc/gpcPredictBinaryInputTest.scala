package dk.gp.gpc

import org.junit._
import Assert._
import breeze.linalg._
import java.io.File
import dk.gp.cov.CovSEiso
import breeze.numerics._

class gpcPredictBinaryInputTest {

  val data = csvread(new File("src/test/resources/gpc/gpc_binary_input.csv"), skipLines = 1)
  val x: DenseMatrix[Double] = data(::, 0 to 0)
  val y = data(::, 1)

  val covFunc = CovSEiso()
  val covFuncParams = DenseVector(log(1), log(0.000001)) //log sf, logEll
  val mean = 0

  @Test def test = {

    val model = GpcModel(x, y, covFunc, covFuncParams, mean)

    val predicted = gpcPredict(x, model)

    assertEquals(0.7728, predicted(1), 0.0001)
    assertEquals(0.2272, predicted(13), 0.0001)
    println(predicted.size)
    println(predicted.map(x => "%.4f".format(x)))
  }
}