package dk.gp.cogp.svi.v

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import dk.gp.math.MultivariateGaussian
import dk.gp.cogp.model.CogpModelParams
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.svi.v.calcLBGradVEta1
import dk.gp.cogp.svi.v.calcLBGradVEta2
import dk.gp.cogp.svi.LBState

/**
 * Stochastic update for the parameters (mu,S) of p(v|y)
 *
 * Nguyen et al. Collaborative Multi-output Gaussian Processes, 2014
 */

object stochasticUpdateV {

  def apply(i: Int,  l: Double, lbState: LBState, y: DenseMatrix[Double],
            kXZ: DenseMatrix[Double], kZZ: DenseMatrix[Double]): MultivariateGaussian = {

    val m = lbState
    
    //natural parameters theta
    val theta1 = inv(m.v(i).v) * m.v(i).m
    val theta2 = -0.5 * inv(m.v(i).v)
   
    val naturalGradEta1 = calcLBGradVEta1(m.beta(i), m.w(i, ::).t, kZZ, kXZ, m.v(i).v, y(::, i), m.v(i).m, m.u)
    val naturalGradEta2 = calcLBGradVEta2(m.beta(i),m.v(i).v, kZZ, kXZ)

    val newTheta1 = theta1 + l * naturalGradEta1
    val newTheta2 = theta2 + l * naturalGradEta2

    val newS = -0.5 * inv(newTheta2)
    val newM = newS * newTheta1
    MultivariateGaussian(newM, newS)
  }
}