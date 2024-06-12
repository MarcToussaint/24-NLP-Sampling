#pragma once

#include <Core/util.h>
#include <Core/array.h>
#include <Algo/ann.h>

//===========================================================================

struct DataMetrics{
  const arr& D;

  DataMetrics(const arr& data) : D(data){}

  arr sequentialDistance();

  arr sequentialMinimalSpanningTree(double coeff=1.);

  arr nearestHistograms(uint kNearest, uint bins, double maxDist=.1);

  double EarthMoverDistance(const arr& Dref);
};


//===========================================================================

arr histogram(const arr& data, const arr& minMax, uint bins);
