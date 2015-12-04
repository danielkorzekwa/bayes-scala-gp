package dk.gp.hgpc.factorgraph2

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import dk.bayes.factorgraph2.api.calibrate
import dk.bayes.factorgraph2.factor.CanonicalGaussianFactor
import dk.bayes.factorgraph2.factor.CanonicalLinearGaussianFactor
import dk.bayes.factorgraph2.factor.StepFunctionFactor
import dk.bayes.factorgraph2.variable.BernVariable
import dk.bayes.factorgraph2.variable.CanonicalGaussianVariable
import dk.bayes.math.gaussian.canonical.CanonicalGaussian
import dk.gp.gp.ConditionalGPFactory
import dk.gp.hgpc.TestCovFunc
import dk.gp.hgpc.getHgpcTestData
import dk.bayes.math.gaussian.canonical.DenseCanonicalGaussian

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

    val uVariable = CanonicalGaussianVariable()

    val taskVariables = taskIds.map(taskId => CanonicalGaussianVariable())

    val taskYVariables: Array[Array[BernVariable]] = taskIds.map { taskId =>

      val idx = x(::, 0).findAll { x => x == taskId }
      val taskY = y(idx).toDenseVector

      val yVariables = y.toArray.map(y => BernVariable(y))

      yVariables
    }

    /**
     * Create factors
     */
    val uFactor = CanonicalGaussianFactor(uVariable, meanU, covU)

    val taskFactors = taskIds.zipWithIndex.map {
      case (taskId, taskIndex) =>
        val idx = x(::, 0).findAll { x => x == taskId }
        val taskX = x(idx, ::).toDenseMatrix
        val (a, b, v) = ConditionalGPFactory(u, covFunc, covFuncParams, mean).create(taskX)
        CanonicalLinearGaussianFactor(uVariable, taskVariables(taskIndex), a, b, v)
    }

    val taskYFactors = taskIds.zipWithIndex.map {
      case (taskId, taskIndex) =>
        val idx = x(::, 0).findAll { x => x == taskId }
        val taskY = y(idx).toDenseVector

        val yFactors = taskY.toArray.zipWithIndex.map {
          case (y, i) =>
            StepFunctionFactor(taskVariables(taskIndex), taskYVariables(taskIndex)(i), taskY.size, i, v = 1)
        }

        yFactors
    }

    /**
     * Update variables
     */
    uVariable.update()
    taskVariables.foreach(_.update())
    taskYVariables.foreach(_.foreach(_.update()))

    /**
     * Calibration
     */

    var calibrated = false
    def calibrateStep() = {

      val beforeUMarginal = uVariable.get()

      taskFactors.zipWithIndex.foreach {
        case (taskFactor, taskIndex) =>

          taskFactor.updateMsgV2()

          taskVariables(taskIndex).update()

          taskYFactors(taskIndex).foreach { taskYFactor => taskYFactor.updateMsgV1() }

          taskVariables(taskIndex).update()

          taskFactor.updateMsgV1()

          uVariable.update()

      }

      calibrated = CanonicalGaussian.isIdentical(beforeUMarginal, uVariable.get, 1e-3)
    }

    val (calib, iter) = calibrate(calibrateStep, 100, calibrated)

    assertTrue(calib)
    assertEquals(9, iter)

    //println(uVariable.get.asInstanceOf[DenseCanonicalGaussian].mean)

    assertEquals(-1.7049183, uVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(0), 0.00001)
    assertEquals(-1.26784, uVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(1), 0.00001)
    assertEquals(-6.40419, uVariable.get.asInstanceOf[DenseCanonicalGaussian].mean(2), 0.00001)
  }
}