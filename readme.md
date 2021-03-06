# Experiments!

[![License is Unlicense](http://img.shields.io/badge/license-Unlicense-blue.svg?style=flat)](http://unlicense.org/)

Platform | Build Status |
-------- | ------------ |
Visual Studio 2015 | [![Build status](https://ci.appveyor.com/api/projects/status/t1v586iy881ptlql?svg=true)](https://ci.appveyor.com/project/ddiakopoulos/sandbox)

A collection of C++11 classes and GLSL shaders united under the arbitrarily chosen namespace "avl" (short for Anvil). 

## Project Structure

  * `virtual-reality\` - OpenVR sandbox implementing a forward renderer.
  * `light-transport\` - A half-finished educational pathtracer started during a vacation in 2016.
  * `incubator.vcxproj` - lib-incubator, shared code between public & private projects.
  * `examples\` - Tests of functionality in lib-incubator. Not likely to compile if you are not me. 

## Recent

* Game-Dev, Procedural Geometry, and Simulation Algorithms
  * Ballistic trajectories - firing solutions for static and dynamic targets (i.e. tower defense)
  * Parabolic pointer - the computational geometry impl of a teleportation mechanic for VR
  * Parallel transport frames (PTF) - useful for creating tube geometries and virtual camera trajectories
  * Supershapes - modeling framework for natural forms based on the Gielis superformula (patented, unfortunately)
  * Library of procedural meshes (cube, sphere, cone, torus, capsule, etc)
  * Simplex noise generator (flow, derivative, curl, worley, fractal, and more)
  * 2D & 3D poisson disk distribution generator
  * Reaction-diffusion CPU simulation (Gray-Scott)
  * Kmeans clustering of pointclouds
  * Quickhull of pointclouds
* Rendering
  * Lightweight, header-only opengl wrapper (GL_API.hpp)
  * Decal projection on arbitrary meshes
  * Spherical environment mapping (matcap shading)
  * Billboarded triangle mesh line renderer
  * Procedural sky dome with Preetham and Hosek-Wilkie models
  * Island terrain with a simple water shader (waves, reflections)
* App-Dev
  * GLSL shader program hot-reload (via efsw library)
  * 2D resolution-independent layout system for screen debug output (textures, etc)
  * Frame capture => GIF export
  * Dear-Imgui Bindings
* DSP Algorithms & Filters
  * One-euro, complimentary, and exponential filters for 1D data
  * Ad-hoc statistical analysis (variance, skewness, kurtosis, etc)
* Misc
  * `libincubator` incorporates a considerable amount of public-domain code adapted for `linalg.h` types or readability/usability (like various lock-free queues). See `COPYING` for details.