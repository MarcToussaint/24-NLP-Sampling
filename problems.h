#include <Kin/kin.h>
#include <KOMO/komo.h>
#include <KOMO/manipTools.h>

//===========================================================================

struct Problem{
  rai::Configuration C;
  std::shared_ptr<ManipulationModelling> manip;
  KOMO komo;
  std::shared_ptr<NLP> nlp;
};

std::shared_ptr<Problem> loadProblem(str problem, uint dim);

//===========================================================================

struct BoxNLP : NLP {
  BoxNLP(uint dim);
  void evaluate(arr& phi, arr& J, const arr& x);
};

//===========================================================================

struct ModesNLP : NLP {
  arr cen;
  arr radii;

  ModesNLP(uint dim, uint k=5);
  void evaluate(arr& phi, arr& J, const arr& x);
};

//===========================================================================

void testPush();
