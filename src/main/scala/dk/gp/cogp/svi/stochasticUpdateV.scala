package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import dk.gp.math.MultivariateGaussian
import breeze.linalg.InjectNumericOps
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.grad.calcLBGradVEta1
import dk.gp.cogp.lb.grad.calcLBGradVEta2
import dk.gp.cogp.model.CogpModel
import breeze.linalg.eig
import breeze.linalg.diag
import dk.gp.cov.CovNoise

/**
 * Stochastic update for the parameters (mu,S) of p(v|y)
 *
 * Nguyen et al. Collaborative Multi-output Gaussian Processes, 2014
 */

object stochasticUpdateV {

  private val learningRate = 1e-2

  def apply(i: Int, lb: LowerBound): MultivariateGaussian = {

    val model = lb.model
    val v = model.h(i).u

    //natural parameters theta
    val vInv = invchol(cholesky(v.v).t)
    val theta1 = vInv * v.m
    val theta2 = -0.5 * vInv

    val naturalGradEta1 = calcLBGradVEta1(i, lb)
    val naturalGradEta2 = calcLBGradVEta2(i, lb)

    val newTheta1 = theta1 + learningRate * naturalGradEta1
    val newTheta2 = theta2 + learningRate * naturalGradEta2

    val newTheta2Eig = eig(newTheta2)
    val invNewTheta2 = newTheta2Eig.eigenvectors * diag(1.0 :/ newTheta2Eig.eigenvalues) * newTheta2Eig.eigenvectors.t

    val newS = -0.5 * invNewTheta2

    val newM = lb.model.h(i).covFunc match {
      case covFunc: CovNoise => v.m
      case _                 => newS * newTheta1
    }

    MultivariateGaussian(newM, newS)
  }
}