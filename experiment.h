#include <Optim/NLP.h>
#include <Core/util.h>

#include "DataMetrics.h"

struct Experiment_Options {
  RAI_PARAM("ex/", rai::String, problem, "none")
  RAI_PARAM("ex/", uint, samples, 20)
  RAI_PARAM("ex/", uint, runs, 10)
  RAI_PARAM("ex/", uint, maxEvals, 100000)
  RAI_PARAM("ex/", int, verbose, 0)
  RAI_PARAM("ex/", bool, calcEMD, false)
  RAI_PARAM("ex/", rai::String, outPath, "")
  //RAI_PARAM("ex/", double, mstsCoeff, 1.)
};

struct Experiment{
  Experiment_Options opt;

  arr data;
  uintA dataEvals;
  std::shared_ptr<DataMetrics> _metrics;
  DataMetrics& metrics();

  double sample(NLP& nlp, int verbose, double alpha_bar);
  void run();

  void plotData(const arr& bounds);
  void plotHistogram(DataMetrics m);

  double EarthMoverDistance();
};
