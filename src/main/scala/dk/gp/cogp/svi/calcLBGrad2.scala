package dk.gp.cogp.svi

import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import scala.math._
import breeze.linalg.inv
import breeze.linalg.sum
import breeze.linalg.diag
import breeze.linalg.cholesky
import dk.gp.cogp.model.CogpModelParams

/**
 * Returns gradient of a lower bound with respect to expectation parameters of q(u).
 * Expectation parameters are eta1 = mean, eta2 = mean^2 + sigma^2.
 *
 * Gradient derivation for eta1:
 * L(m) = L(f(eta1)) = F(x)
 * F'(x) = L'(f(eta1))*f'(eta1) = L'(f(eta1)) = L'(m)
 *
 * similarly for eta2
 * L(S^2) = L(f(eta2) = F(x)
 * F'(x) = L'(f(eta2))*f'(eta2) = L'(f(eta2)) = L'(S^2)
 *
 * Nguyen et al. Collaborative Multi-output Gaussian Processes, 2014, relevant equations 19, 20, 23, 24
 *
 * Useful reading
 * Hensman et al. Gaussian Processes for Big Data, 2013, relevant equations 4, 5, 6, eq below 6
 *
 */
object calcLBGrad2 {

  /**
   * @param Index of Q variable
   * @param beta [Px1]
   * @param w [i x j]
   */
  def apply(j: Int,   modelParams:CogpModelParams, y: DenseMatrix[Double], 
       kXZ:DenseMatrix[Double],kZZ:DenseMatrix[Double]): Tuple2[DenseVector[Double], DenseMatrix[Double]] = {
    
    val m = modelParams
    
    val eta1Grad = calcGradEta1(j, m.beta, m.w,  y,   kZZ, kXZ,modelParams)
    val eta2Grad = calcGradEta2(j, m.beta, m.w,  kZZ, kXZ,modelParams)

    (eta1Grad, eta2Grad)
  }

  private def calcGradEta1(j: Int, beta: DenseVector[Double], w: DenseMatrix[Double], y: DenseMatrix[Double],
                           kZZ: DenseMatrix[Double], kXZ: DenseMatrix[Double],modelParams:CogpModelParams): DenseVector[Double] = {

    val u = modelParams.u
    val v = modelParams.v
    
    val A = kXZ * inv(kZZ)

    val tmp = (0 until beta.size).map { i =>
      val betaVal = beta(i)
      val wVal = w(i, j)

      val othersJIdx = (0 until w.cols).filter(jIndex => jIndex != j)
      val wAm = if (othersJIdx.size > 0) {
        othersJIdx.map { jIndex => w(i, jIndex) * A * u(jIndex).m }.toArray.sum
      } else DenseVector.zeros[Double](y.rows)

      val yVal = y(::, i) - A * v(i).m - wAm

      betaVal * wVal * A.t * yVal
    }.reduceLeft((total, x) => total + x)

    val eta1Grad = tmp - inv(u(j).v) * u(j).m
    eta1Grad
  }

  private def calcGradEta2(j: Int, beta: DenseVector[Double], w: DenseMatrix[Double],  kZZ: DenseMatrix[Double], kXZ: DenseMatrix[Double],modelParams:CogpModelParams): DenseMatrix[Double] = {

    val u = modelParams.u
    
    val A = kXZ * inv(kZZ)
    val tmp = (0 until beta.size).map { i =>
      val betaVal = beta(i)
      val w2 = pow(w(i, j), 2)
      betaVal * w2 * A.t * A
    }.reduceLeft((total, x) => total + x)

    val lambda = inv(kZZ) + tmp
    
    val eta2Grad = 0.5 * inv(u(j).v) - 0.5 * lambda
  
    eta2Grad

  }

  implicit class DenseVectorOps(seq: Array[DenseVector[Double]]) {
    def sum(): DenseVector[Double] = seq match {
      case Array(x) => x
      case seq      => seq.reduceLeft((total, x) => total + x)
    }
  }
}