package dk.gp.gpc.factorgraph2

case class Message[T]() {

  private var m: T = _

  def get():T = m
  def set(m: T) = { this.m = m }
  
 
}
 