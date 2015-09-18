package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.cogp.model.CogpModel
import dk.gp.cogp.svi.stochasticUpdateCogpModel
import breeze.linalg.DenseVector
import dk.gp.cogp.lb.LowerBound
import com.typesafe.scalalogging.slf4j.LazyLogging
import dk.gp.cogp.lb.calcLBLoglik

object cogpTrain extends LazyLogging{

  def apply(x: DenseVector[Double], y: DenseMatrix[Double], model: CogpModel, iterNum: Int):CogpModel = {
    cogpTrain(x.toDenseMatrix.t,y,model,iterNum)
  }
  
  def apply(x: DenseMatrix[Double], y: DenseMatrix[Double], model: CogpModel, iterNum: Int): CogpModel = {
    val finalModel = (0 until iterNum).foldLeft(model) { 
      case (currModel, iter) => 
        val newModel = stochasticUpdateCogpModel(currModel, x, y)
          val lb = LowerBound(newModel, x, y)
          logger.info("llh=%.3f".format(calcLBLoglik(lb)))
          newModel
      }
    finalModel
  }
}