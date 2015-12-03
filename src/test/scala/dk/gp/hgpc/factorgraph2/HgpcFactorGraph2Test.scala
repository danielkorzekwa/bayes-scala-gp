package dk.gp.hgpc.factorgraph2

import org.junit._
import Assert._
import dk.gp.hgpc.getHgpcTestData
import breeze.linalg.DenseVector
import dk.gp.hgpc.TestCovFunc
import dk.gp.gpc.factorgraph2.GaussianVariable
import dk.gp.gpc.factorgraph2.BernVariable
import dk.gp.gpc.factorgraph2.BernVariable
import breeze.linalg.DenseMatrix
import breeze.linalg._
import breeze.numerics._
import dk.gp.gpc.factorgraph2.MultivariateGaussianFactor
import dk.gp.gpc.factorgraph2.LinearGaussianFactor
import dk.gp.gpc.factorgraph2.MvnGaussianThresholdFactor
import dk.gp.gp.ConditionalGPFactory

class HgpcFactorGraph2Test {

  val (x, y, u) = getHgpcTestData()

  val covFunc = TestCovFunc()
  val covFuncParams = DenseVector(1.219809, 0.052334, 0.171199) //log sf, logEllx1, logEllx2
  val mean = -2.8499

  val covU = covFunc.cov(u, u, covFuncParams) + DenseMatrix.eye[Double](u.rows) * 1e-7
  val meanU = DenseVector.zeros[Double](u.rows) + mean

  val taskIds = x(::, 0).toArray.distinct

  @Test def test: Unit = {

    /**
     * Create variables
     */

    val uVariable = GaussianVariable()

    val taskVariables = taskIds.map(taskId => GaussianVariable())

    val taskYVariables: Array[Array[BernVariable]] = taskIds.map { taskId =>

      val idx = x(::, 0).findAll { x => x == taskId }
      val taskY = y(idx).toDenseVector

      val yVariables = y.toArray.map(y => BernVariable(y))

      yVariables
    }

    /**
     * Create factors
     */
    val uFactor = MultivariateGaussianFactor(uVariable,meanU, covU)

    val taskFactors =  taskIds.zipWithIndex.map { case(taskId,taskIndex) =>
      val idx = x(::, 0).findAll { x => x == taskId }
      val taskX = x(idx, ::).toDenseMatrix
      val (a, b, v) = ConditionalGPFactory(u, covFunc, covFuncParams, mean).create(taskX)
      LinearGaussianFactor(uVariable,taskVariables(taskIndex),a, b, v)
    }

    val taskYFactors = taskIds.zipWithIndex.map { case(taskId,taskIndex) =>
      val idx = x(::, 0).findAll { x => x == taskId }
      val taskY = y(idx).toDenseVector

      val yFactors = y.toArray.zipWithIndex.map {
        case (y, i) =>
          MvnGaussianThresholdFactor(taskVariables(taskIndex),taskYVariables(taskIndex)(i),taskY.size, i, v = 1)
      }

      yFactors
    }

    /**
     * Update variables
     */
    uVariable.update()
    taskVariables.foreach(_.update())
    taskYVariables.foreach(_.foreach(_.update()))

  }
}