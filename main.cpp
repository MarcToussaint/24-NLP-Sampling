#include "experiment.h"

#include <Core/util.h>

//===========================================================================

int main(int argc,char** argv){
  rai::initCmdLine(argc,argv);

  rnd.clockSeed();

//    testing_walker(*P->nlp, samples, verbose, 1.); return 0;

//  testPush();

  Experiment E;
  E.run();

  return 0;
}



