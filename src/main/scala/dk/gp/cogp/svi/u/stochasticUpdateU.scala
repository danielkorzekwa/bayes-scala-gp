package dk.gp.cogp.svi.u

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.inv
import breeze.linalg.cholesky
import dk.gp.math.MultivariateGaussian
import dk.gp.cogp.CogpModel
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound

/**
 * Stochastic update for the parameters (mu,S) of q(u|y)
 *
 * Nguyen et al. Collaborative Multi-output Gaussian Processes, 2014
 */
object stochasticUpdateU {

  private val learningRate = 1e-2

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
  def apply(j: Int, lowerBound:LowerBound,model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): MultivariateGaussian = {

    val z = model.g(j).z
    val kXZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams)
    val kZZ = model.g(j).covFunc.cov(z, z, model.g(j).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

    val u = model.g.map(_.u)
    //natural parameters theta
    //val vInv = invchol(cholesky(u(j).v).t) //@TODO use chol with jitter for matrix inverse
    val vInv = inv(u(j).v)
    val theta1 = vInv * u(j).m
    val theta2 = -0.5 * vInv

    /**
     * Natural gradient with respect to natural parameter is just a standard gradient with respect to expectation parameters.
     * Thus no need for inverse of Fisher information matrix. Sweet.
     *  Hensman et al. Gaussian Processes for Big Data, 2013
     */
    val naturalGradEta1 = calcLBGradUEta1(j, lowerBound,model,x,y)
    val naturalGradEta2 = calcLBGradUEta2(j, lowerBound,model,x,y)

    val newTheta1 = theta1 + learningRate * naturalGradEta1
    val newTheta2 = theta2 + learningRate * naturalGradEta2

   // val newTheta2Inv = invchol(cholesky(newTheta2).t)
    val newTheta2Inv = inv(newTheta2) //use invchol with jitter
    
    val newS = -0.5 * newTheta2Inv
    val newM = newS * newTheta1
    MultivariateGaussian(newM, newS)

  }
}