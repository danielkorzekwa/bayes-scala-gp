package dk.gp.cogp

import org.junit.Test
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.numerics.log
import dk.gp.cov.CovSEiso
import breeze.linalg._
import java.io.File

class predictTest {

   val data = csvread(new File("src/test/resources/cogp/cogp_no_missing_points.csv"))(0 to 4,::)
  val x = data(::,0).toDenseMatrix.t
  val y = data(::,1 to 2)
  
  //val x: DenseMatrix[Double] = DenseVector.rangeD(-10, 10, 1).toDenseMatrix.t
  //val y = DenseVector.horzcat(DenseVector.zeros[Double](x.size) + 1d, DenseVector.zeros[Double](x.size) + 2d)

  val covFuncParams = DenseVector(log(1), log(1))
  val covFunc = CovSEiso()

  @Test def test = {

    val model = cogp(x, y, covFunc, covFuncParams)

    val loglik = calcLBLoglik(model.u,model.v,model.beta,model.w,model.kZZ,model.kXZ,model.kXXDiag,y)
    println("LB loglik=" + loglik)

    val predictedY = predict(x, model)

    println(predictedY)
  }
}