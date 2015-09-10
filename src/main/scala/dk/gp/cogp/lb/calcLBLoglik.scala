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
import dk.gp.cogp.CogpModel
import dk.gp.cov.utils.covDiag
import dk.gp.math.invchol
import breeze.linalg.Axis

object calcLBLoglik {

  def apply(lowerBound: LowerBound, cogpModel: CogpModel, x: DenseMatrix[Double], y: DenseMatrix[Double]): Double = {

    val gArray = cogpModel.g
    val hArray = cogpModel.h
    val beta = cogpModel.beta
    val w = cogpModel.w

    val qTerm = (0 until gArray.size).map { j =>

      val kZZ = lowerBound.kZZj(j)
      val kZZinv = lowerBound.kZZjInv(j)

      val u = gArray(j).u
      val qTerm_j = 0.5 * (logdet(kZZ)._2 + logdet(inv(u.v))._2) + 0.5 * trace(kZZinv * (u.m * u.m.t + u.v)) //is it better to compute log det using chol decomposition? look at cogp impl and alg 2.1 of Rasmussen book
      qTerm_j
    }.sum

    val pTerm = (0 until hArray.size).map { i =>

      val z = cogpModel.h(i).z
      val kZZ2 = cogpModel.h(i).covFunc.cov(z, z, cogpModel.h(i).covFuncParams) + 1e-10 * DenseMatrix.eye[Double](z.size)
      val kXZ2 = cogpModel.h(i).covFunc.cov(z, z, cogpModel.h(i).covFuncParams)
      val kZX2 = kXZ2.t
      val kZZ2inv = invchol(cholesky(kZZ2).t)
      val Ai2 = kXZ2 * kZZ2inv
      val lambdaI = Ai2.t * Ai2

      val kXXDiag_i = covDiag(x, cogpModel.h(i).covFunc, cogpModel.h(i).covFuncParams)

      val kTildeDiagSum_i = sum(kXXDiag_i - diag(kXZ2 * kZZ2inv * kZX2))

      val v = hArray(i).u
      val pTerm_i = 0.5 * (logdet(kZZ2)._2 - logdet(v.v)._2) + 0.5 * trace(kZZ2inv * ((v.m * v.m.t + v.v)))

      val traceQTerm = (0 until gArray.size).map { j =>

        val z = cogpModel.g(j).z

        //@TODO use (x,z) instead of (z,z), similarly in other places in the project
        val kXZ = lowerBound.kXZj(j)
        val kZZinv = lowerBound.kZZjInv(j)

        val Aj = kXZ * kZZinv
        val lambdaJ = Aj.t * Aj
        val u = gArray(j).u
        pow(w(i, j), 2) * trace(u.v * lambdaJ)
      }.sum

      val kTildeQTerm = (0 until gArray.size).map { j =>

        val kXZ = lowerBound.kXZj(j)
        val kZZ = lowerBound.kZZj(j)
        val kZX = kXZ.t
        val kXXDiag = covDiag(x, cogpModel.g(j).covFunc, cogpModel.g(j).covFuncParams)

        val kZZinv = lowerBound.kZZjInv(j)

        //@TODO performance improvement:
        /**
         * trace(ABC) = trace(CAB) or trace(ABC) = sum(sum(ab.*c',2))
         * https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/MatrixCookBook.pdf,
         * https://github.com/trungngv/cogp/blob/master/libs/util/diagProd.m
         */
        val kTildeDiagSum = sum(kXXDiag - diag(kXZ * kZZinv * kZX))
      //   val kTildeDiagSum = sum(kXXDiag - sum((kXZ*kZZinv):*kZX,Axis._1)) 
     //    val kTildeDiagSum = sum(kXXDiag - trace(kZX*kXZ*kZZinv))
        pow(w(i, j), 2) * kTildeDiagSum
      }.sum

      val tracePTerm = beta(i) * 0.5 * trace(v.v * lambdaI)
      val kTildePTerm = 0.5 * beta(i) * kTildeDiagSum_i

      val wAm = (0 until gArray.size).foldLeft(DenseVector.zeros[Double](x.rows)) { (wAm, j) =>

        val kXZ = lowerBound.kXZj(j)
        val kZZ = lowerBound.kZZj(j)
        val kZZinv = lowerBound.kZZjInv(j)

        val Aj = kXZ * kZZinv

        wAm + w(i, j) * Aj * gArray(j).u.m
      }

      val yTerm = y(::, i) - wAm - Ai2 * hArray(i).u.m

      val logNTerm = -0.5 * y.rows * log(2 * Pi / beta(i)) - 0.5 * beta(i) * sum(pow(yTerm, 2))


      val pTerm = logNTerm - pTerm_i - tracePTerm - kTildePTerm - 0.5 * beta(i) * traceQTerm - 0.5 * beta(i) * kTildeQTerm
      pTerm
      
     
    }.sum

    -qTerm + pTerm
  }

}