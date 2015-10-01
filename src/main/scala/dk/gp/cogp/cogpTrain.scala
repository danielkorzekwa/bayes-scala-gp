package dk.gp.cogp

import breeze.linalg.DenseMatrix
import dk.gp.cogp.model.CogpModel
import dk.gp.cogp.svi.stochasticUpdateCogpModel
import breeze.linalg.DenseVector
import dk.gp.cogp.lb.LowerBound
import com.typesafe.scalalogging.slf4j.LazyLogging
import dk.gp.cogp.lb.calcLBLoglik
import dk.gp.cogp.model.Task

object cogpTrain extends LazyLogging{
  
  def apply(tasks:Array[Task], model: CogpModel, iterNum: Int): CogpModel = {
    val finalModel = (0 until iterNum).foldLeft(model) { 
      case (currModel, iter) => 
        val newModel = stochasticUpdateCogpModel(currModel,tasks)
          val lb = LowerBound(newModel, tasks)
          logger.info("iter=%d/%d, llh=%.3f".format(iter,iterNum,calcLBLoglik(lb)))
          newModel
      }
    finalModel
  }
}