# Algorithms for Gaussian Processes

[![Build Status](https://travis-ci.org/danielkorzekwa/bayes-scala-gp.svg)](https://travis-ci.org/danielkorzekwa/bayes-scala-gp)

List of algorithms:
* [Collaborative multi-output Gaussian Process](https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/doc/cogp/cogp.md)
* [Sparse Gaussian Process Regression](https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/doc/sgpr/sgpr.md)
* [Gaussian Process Regression](https://github.com/danielkorzekwa/bayes-scala-gp/blob/master/doc/gpr/gpr.md)

This project is related to a [Bayes-scala](https://github.com/danielkorzekwa/bayes-scala) library.

## How to use it from sbt and maven?

Release and snaphot versions are available for Scala versions 2.10 and 2.11

### Snapshot version

Snapshot artifact is built by a Travis CI and deployed to Sonatype OSS Snapshots repository with every commit to Bayes-scala-gp project. 

With sbt build tool, add to build.sbt config file:

```scala
libraryDependencies += "com.github.danielkorzekwa" %% "bayes-scala-gp" % "0.1-SNAPSHOT"  

resolvers += Resolver.sonatypeRepo("snapshots")
```

With maven build tool, add to pom.xml config file:

```scala
  <repositories>
    <repository>
      <id>oss-sonatype-snapshots</id>
      <name>oss-sonatype-snapshots</name>
      <url>https://oss.sonatype.org/content/repositories/snapshots/</url>
    </repository>
  </repositories>
  
  <dependencies>
    <dependency>
      <groupId>com.github.danielkorzekwa</groupId>
      <artifactId>bayes-scala-gp_2.11</artifactId>
      <version>0.1-SNAPSHOT</version>
    </dependency>
  <dependencies>
```