package dk.gp.cogp.svi

import org.junit._
import Assert._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector

class classicalMomentumTest {

  val epsilon = 0.1
  val mu = 0.9

  /**
   *  Maximise function f(x) = -(x^2 + 2x + 5)
   */
  @Test def test = {
    import breeze.linalg._

    val (theta, thetaDelta) = (1 to 100).foldLeft((DenseVector(1.0).toDenseMatrix, DenseVector(0.0).toDenseMatrix)) { (currState, iterNum) =>

      val (theta, thetaDelta) = currState
      println("current theta/thetaDelta: %.2f/%.2f".format(theta(0, 0), thetaDelta(0, 0)))
      val thetaGrad = -2.0 * theta - 2.0
      val (newTheta, newThetaDelta) = classicalMomentum(theta, thetaDelta, epsilon, mu, thetaGrad)
      (newTheta, newThetaDelta)

    }

    assertEquals(-1.0057, theta(0, 0), 0.0001)
    assertEquals(-0.00437, thetaDelta(0, 0), 0.0001)
  }
}