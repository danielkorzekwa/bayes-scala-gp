package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.inv
import breeze.linalg.cholesky
import dk.gp.math.MultivariateGaussian
import dk.gp.cogp.model.CogpModelParams

/**
 * Stochastic update for the parameters (mu,S) of q(u|y)
 *
 * Nguyen et al. Collaborative Multi-output Gaussian Processes, 2014
 */
object stochasticUpdateU {

  /**
   * @param j Index of Q variable
   * @param beta [P x 1]
   * @param w [P x Q]
   * @param mQ [Z x Q]
   * @param mP [Z x P]
   * @param S [Z x Z]
   * @param kZZ [Z x Z]
   * @param kXZ [X x Z]
   * @param y [X x P]
   * @param l Learning rate
   */
  def apply(j: Int,  l: Double, modelParams:CogpModelParams, y: DenseMatrix[Double],kXZ:DenseMatrix[Double],kZZ:DenseMatrix[Double]): MultivariateGaussian = {

    val u = modelParams.u
    //natural parameters theta
    val theta1 = inv(u(j).v) * u(j).m
    val theta2 = -0.5 * inv(u(j).v)

    /**
     * Natural gradient with respect to natural parameter is just a standard gradient with respect to expectation parameters.
     * Thus no need for inverse of Fisher information matrix. Sweet.
     *  Hensman et al. Gaussian Processes for Big Data, 2013
     */
    val (naturalGradEta1, naturalaGradEta2) = calcLBGrad2(j, modelParams,y,kXZ,kZZ)

    val newTheta1 = theta1 + l * naturalGradEta1
    val newTheta2 = theta2 + l * naturalaGradEta2
  
    val newS = -0.5 * inv(newTheta2)
    val newM = newS * newTheta1
    MultivariateGaussian(newM, newS)

  }
}