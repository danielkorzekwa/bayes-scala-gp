package dk.gp.cogp.lb

import scala.math.Pi
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.InjectNumericOps
import breeze.linalg.cholesky
import breeze.linalg.diag
import breeze.linalg.inv
import breeze.linalg.logdet
import breeze.linalg.sum
import breeze.linalg.trace
import breeze.numerics.log
import breeze.numerics.pow
import dk.gp.cov.utils.covDiag
import dk.gp.math.invchol
import breeze.linalg.Axis

object calcLBLoglik {

  def apply(lowerBound: LowerBound): Double = {

    val beta = lowerBound.model.beta

    val pTerm = (0 until lowerBound.model.h.size).map { i =>

      val pTerm = logNTerm(i, lowerBound) -
        pTerm_i(i, lowerBound) - tracePTerm(i, lowerBound) -
        kTildePTerm(i, lowerBound) -
        0.5 * beta(i) * traceQTerm(i, lowerBound) -
        0.5 * beta(i) * kTildeQTerm(i, lowerBound)
      pTerm

    }.sum

    -qTerm(lowerBound) + pTerm
  }

  private def qTerm(lb: LowerBound): Double = {
    val qTerm = (0 until lb.model.g.size).map { j =>

      val kZZ = lb.kZZj(j)
      val kZZinv = lb.kZZjInv(j)

      val u = lb.model.g(j).u
      val qTerm_j = 0.5 * (logdet(kZZ)._2 + logdet(inv(u.v))._2) + 0.5 * trace(kZZinv * (u.m * u.m.t + u.v)) //is it better to compute log det using chol decomposition? look at cogp impl and alg 2.1 of Rasmussen book
      qTerm_j
    }.sum

    qTerm
  }
  private def pTerm_i(i: Int, lb: LowerBound): Double = {

    val kZZ2 = lb.kZZi(i)
    val kZZ2inv = lb.kZZiInv(i)

    val v = lb.model.h(i).u

    val pTerm_i = 0.5 * (logdet(kZZ2)._2 - logdet(v.v)._2) + 0.5 * trace(kZZ2inv * ((v.m * v.m.t + v.v)))

    pTerm_i
  }

  private def logNTerm(i: Int, lb: LowerBound): Double = {

    val Ai = lb.calcAi(i)
 
    val y = lb.yi(i)
    
    val yTerm = y - wAm(i, lb) - Ai * lb.model.h(i).u.m
    val logNTerm = -0.5 * y.size * log(2 * Pi / lb.model.beta(i)) - 0.5 * lb.model.beta(i) * sum(pow(yTerm, 2))

    logNTerm
  }

  private def kTildePTerm(i: Int, lb: LowerBound): Double = {
    val kXZ2 = lb.kXZi(i)
    val kZX2 = kXZ2.t
    val kZZ2inv = lb.kZZiInv(i)

    val kXXDiag_i = covDiag(lb.x, lb.model.h(i).covFunc, lb.model.h(i).covFuncParams)

    val kTildeDiagSum_i = sum(kXXDiag_i - diag(kXZ2 * kZZ2inv * kZX2))

    val kTildePTerm = 0.5 * lb.model.beta(i) * kTildeDiagSum_i
    kTildePTerm
  }

  private def tracePTerm(i: Int, lb: LowerBound): Double = {

    val Ai = lb.calcAi(i)
    val lambdaI = Ai.t * Ai

    val v = lb.model.h(i).u
    lb.model.beta(i) * 0.5 * trace(v.v * lambdaI)
  }

  private def traceQTerm(i: Int, lb: LowerBound): Double = {
    val traceQTerm = (0 until lb.model.g.size).map { j =>

      val Aj = lb.calcAj(j)
      val lambdaJ = Aj.t * Aj
      val u = lb.model.g(j).u
      pow(lb.model.w(i, j), 2) * trace(u.v * lambdaJ)
    }.sum

    traceQTerm
  }

  private def kTildeQTerm(i: Int, lb: LowerBound): Double = {
    val kTildeQTerm = (0 until lb.model.g.size).map { j =>

      val kXZ = lb.kXZj(j)
      val kZZ = lb.kZZj(j)
      val kZX = kXZ.t
      val kXXDiag = covDiag(lb.x, lb.model.g(j).covFunc, lb.model.g(j).covFuncParams)

      val kZZinv = lb.kZZjInv(j)

      //@TODO performance improvement:
      /**
       * trace(ABC) = trace(CAB) or trace(ABC) = sum(sum(ab.*c',2))
       * https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/MatrixCookBook.pdf,
       * https://github.com/trungngv/cogp/blob/master/libs/util/diagProd.m
       */
      val kTildeDiagSum = sum(kXXDiag - diag(kXZ * kZZinv * kZX))
      //   val kTildeDiagSum = sum(kXXDiag - sum((kXZ*kZZinv):*kZX,Axis._1)) 
      //    val kTildeDiagSum = sum(kXXDiag - trace(kZX*kXZ*kZZinv))
      pow(lb.model.w(i, j), 2) * kTildeDiagSum
    }.sum

    kTildeQTerm
  }

}