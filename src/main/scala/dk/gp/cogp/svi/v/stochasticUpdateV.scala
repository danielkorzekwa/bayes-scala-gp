package dk.gp.cogp.svi.v

import breeze.linalg.DenseMatrix
import breeze.linalg.inv
import dk.gp.math.MultivariateGaussian
import breeze.linalg.InjectNumericOps
import dk.gp.cogp.CogpModel
import breeze.linalg.cholesky
import dk.gp.math.invchol
import dk.gp.cogp.lb.LowerBound

/**
 * Stochastic update for the parameters (mu,S) of p(v|y)
 *
 * Nguyen et al. Collaborative Multi-output Gaussian Processes, 2014
 */

object stochasticUpdateV {

  private val learningRate = 1e-2

  def apply(i: Int, lowerBound:LowerBound,model: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): MultivariateGaussian = {

    val z = x
    val kXZ = model.g.head.covFunc.cov(z, z, model.g.head.covFuncParams)
    val kZZ = model.g.head.covFunc.cov(z, z, model.g.head.covFuncParams) + 1e-10 * DenseMatrix.eye[Double](x.size)

    val m = model

    val u = model.g.map(_.u)
    val v = model.h.map(_.u)

    //natural parameters theta
    val vInv = invchol(cholesky(v(i).v).t)
    val theta1 = vInv * v(i).m
    val theta2 = -0.5 * vInv

    val naturalGradEta1 = calcLBGradVEta1(i, lowerBound,model, x, y)
    val naturalGradEta2 = calcLBGradVEta2(i, lowerBound,model, x, y)

    val newTheta1 = theta1 + learningRate * naturalGradEta1
    val newTheta2 = theta2 + learningRate * naturalGradEta2

    val newS = -0.5 * inv(newTheta2) //@TODO use invchol with jitter

    //@TODO following Nguyen, why is that: a bit of hack to allow h_i to be input-dependent noise
    //val newM = newS * newTheta1
    val newM = v(i).m
    MultivariateGaussian(newM, newS)
  }
}