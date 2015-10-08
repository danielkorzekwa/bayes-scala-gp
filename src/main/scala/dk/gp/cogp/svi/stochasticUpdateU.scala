package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.inv
import breeze.linalg.cholesky
import dk.gp.math.MultivariateGaussian
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound
import dk.gp.cogp.lb.grad.calcLBGradUEta1
import dk.gp.cogp.lb.grad.calcLBGradUEta2
import dk.gp.cogp.model.CogpModel
import breeze.linalg.eig
import breeze.linalg.diag
import breeze.linalg._

/**
 * Stochastic update for the parameters (mu,S) of q(u|y)
 *
 * Nguyen et al. Collaborative Multi-output Gaussian Processes, 2014
 */
object stochasticUpdateU {

  private val learningRate = 1e-2

  def apply(j: Int, lb: LowerBound): MultivariateGaussian = {

    val u = lb.model.g.map(_.u)
    //natural parameters theta
    val vInv = invchol(cholesky(u(j).v).t) //@TODO use chol with jitter for matrix inverse
    val theta1 = vInv * u(j).m
    val theta2 = -0.5 * vInv

    /**
     * Natural gradient with respect to natural parameter is just a standard gradient with respect to expectation parameters.
     * Thus no need for inverse of Fisher information matrix. Sweet.
     *  Hensman et al. Gaussian Processes for Big Data, 2013
     */
    val naturalGradEta1 = calcLBGradUEta1(j, lb)
    
    val naturalGradEta2 = calcLBGradUEta2(j, lb)

    val newTheta1 = theta1 + learningRate * naturalGradEta1
    val newTheta2 = theta2 + learningRate * naturalGradEta2

    val newTheta2Eig = eig(newTheta2)
    val newTheta2Inv = newTheta2Eig.eigenvectors * diag(1.0 :/ newTheta2Eig.eigenvalues) * newTheta2Eig.eigenvectors.t

    val newS = -0.5 * newTheta2Inv
    val newM = newS * newTheta1
   
    MultivariateGaussian(newM, newS)

  }
}