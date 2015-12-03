package dk.gp.gpc.factorgraph2

import scala.collection.mutable.ListBuffer

trait Variable[T] {

  private var value: T = _

  private var messages: ListBuffer[Message[T]] = ListBuffer()

  def getMessages(): Seq[Message[T]] = messages
  def addMessage(m: Message[T]) = messages += m

  def get(): T = value

  def update() = value = calcVariable()

  def calcVariable(): T
}