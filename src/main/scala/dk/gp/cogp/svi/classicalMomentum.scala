package dk.gp.cogp.svi

import breeze.linalg.DenseMatrix

/**
 * Ilya Sutskever et al. On the importance of initialization and momentum in deep learning, 2013 (eq. 1, 2)
 * http://jmlr.org/proceedings/papers/v28/sutskever13.pdf
 */
object classicalMomentum {

  /**
   * @param theta
   * @param thetaDelta
   * @param epsilon learning rate > 0
   * @param mu the momentum coefficient between [0,1]
   * @param thetaGrad
   * Returns (newTheta,newThetaDelta)
   */
  def apply(theta: DenseMatrix[Double], thetaDelta: DenseMatrix[Double], epsilon: Double, mu: Double, thetaGrad: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val newDelta = mu * thetaDelta + epsilon * thetaGrad // gradient is added not removed, as we maximise instead of minimising the function
    val newTheta = theta + newDelta

    (newTheta, newDelta)
  }

}