package dk.gp.mtgpc.util

import dk.gp.gpc.GpcModel
import dk.gp.gpc.util.GpcFactorGraph
import dk.gp.mtgpc.MtgpcModel

case class MtgpcFactorGraph(var model: MtgpcModel) {

  val taskFactorGraphs = createGpcFactorGraphs()

  private def createGpcFactorGraphs(): Seq[GpcFactorGraph] = {

    val taskIds = model.x(::, 0).toArray.distinct

    val gpcFactorGraphs = taskIds.map { cId =>
      val idx = model.x(::, 0).findAll { x => x == cId }
      val taskX = model.x(idx, ::).toDenseMatrix
      val taskY = model.y(idx).toDenseVector

      val taskModel = GpcModel(taskX, taskY, model.covFunc, model.covFuncParams, model.gpMean)
      val gpcFactorGraph = GpcFactorGraph(taskModel)
      gpcFactorGraph
    }
    gpcFactorGraphs
  }
}