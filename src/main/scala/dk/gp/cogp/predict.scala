package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.math.UnivariateGaussian
import breeze.linalg.DenseVector
import breeze.linalg.inv
import breeze.numerics._
import dk.gp.cogp.model.CogpModel

object predict {

  
 def apply(sMatrix: DenseMatrix[Double], model: CogpModel):DenseMatrix[UnivariateGaussian] = {
   
   val predictedY = (0 until sMatrix.rows).map { sIndex =>
      val s = sMatrix(sIndex, ::).t
      predict(s,  model)
    }

    DenseVector.horzcat[UnivariateGaussian](predictedY: _*).t
 }
 
  def apply(s: DenseVector[Double],  model: CogpModel): DenseVector[UnivariateGaussian] = {

    val u = model.modelParams.u
    val v = model.modelParams.v
    val w = model.modelParams.w
    val beta = model.modelParams.beta
    
    val kSS = model.covFunc.cov(s.toDenseMatrix.t, s.toDenseMatrix.t,model.covFuncParams)(0,0)
    
    val kSZ = model.covFunc.cov(s.toDenseMatrix.t, model.z,model.covFuncParams)(0, ::)
    val kZZinv = inv(model.kZZ)

    val w2 = pow(w, 2)

    val gM = DenseVector(u.map(u => kSZ * kZZinv * u.m))
    val gVar = DenseVector(u.map(u => kSS - kSZ * (kZZinv - kZZinv * u.v * kZZinv) * kSZ.t))

    val hM = v.map(v => kSZ * kZZinv * v.m)
    val hVar = DenseVector(v.map(v => kSS - kSZ * (kZZinv - kZZinv * v.v * kZZinv) * kSZ.t))

    val predictedOutputs = (0 until beta.size).map { i =>

      val outputM = w(i, ::) * gM + hM(i)
      val outputVar = w2(i, ::) * gVar + hVar(i)

      UnivariateGaussian(outputM, outputVar)

    }.toArray

    DenseVector(predictedOutputs)
  }
}