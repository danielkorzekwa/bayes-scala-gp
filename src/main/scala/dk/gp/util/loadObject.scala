package dk.gp.util

import scala.reflect.ClassTag
import java.io.FileInputStream
import com.twitter.chill.ScalaKryoInstantiator
import com.esotericsoftware.kryo.io.Input

object loadObject {

  def apply[T: ClassTag](file: String): T = {

    val instantiator = new ScalaKryoInstantiator()
    instantiator.setRegistrationRequired(false)

    val kryo = instantiator.newKryo()

    val fileIn = new FileInputStream(file)
    val input = new Input(fileIn)

    val deser = kryo.readObject(input, implicitly[ClassTag[T]].runtimeClass)
    deser.asInstanceOf[T]
  }
}