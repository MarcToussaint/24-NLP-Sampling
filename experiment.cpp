#include "experiment.h"

#include "DataMetrics.h"
#include "problems.h"

#include <unistd.h>

#include <Optim/NLP_Sampler.h>

void Experiment::plotData(const arr& bounds){
  //FILE("z.dat") <<data.modRaw();
  str cmd;
  cmd <<"set size ratio -1; set view equal xy\n";
  cmd <<"plot [" <<bounds(0,0) <<':' <<bounds(1,0) <<"][" <<bounds(0,1) <<':' <<bounds(1,1)<<']';
  cmd <<"'samples.dat' us 1:2 w p";
  gnuplot(cmd);
  rai::wait();
}

double Experiment::EarthMoverDistance(){
  rai::FileToken fil("samples.ref.dat");
  if(!fil.exists()) return -1;
  arr Dref;
  fil >>Dref;
  return metrics().EarthMoverDistance(Dref);
}

DataMetrics& Experiment::metrics(){
  if(!_metrics) _metrics = make_shared<DataMetrics>(data);
  return *_metrics;
}

double Experiment::sample(NLP& nlp, int verbose, double alpha_bar){
  NLP_Walker walk(nlp, alpha_bar);

  data.clear();
  dataEvals.clear();
  _metrics.reset();

  for(;;){//restarts
    walk.run(data, dataEvals); //one run: restart+downhill+potentially multiple interior samples

    if(data.d0>=opt.samples) break;
    if(walk.evals>=opt.maxEvals) break;
  }
  CHECK_EQ(data.d0, dataEvals.N, "");

  //cut the data
  if(data.d0>opt.samples){
    data.resizeCopy(opt.samples, data.d1);
    dataEvals.resizeCopy(opt.samples);
  }

  // samples file
  FILE("samples.dat") <<data.modRaw() <<endl;

  cout <<endl;
  LOG(0) <<"evals/sample: " <<double(walk.evals)/double(data.d0);

  if(nlp.featureTypes(-1)==OT_sos){
    LOG(0) <<"Earth Mover Distance: " <<EarthMoverDistance();
  }

  arr MSTS1 = metrics().sequentialMinimalSpanningTree(1.);
  arr MSTS2 = metrics().sequentialMinimalSpanningTree(2.);

  // run file
  ofstream fil("run.dat");
  CHECK_EQ(data.d0, MSTS1.N, "");
  CHECK_EQ(data.d0, MSTS2.N, "");
  for(uint i=0;i<data.d0;i++){
    fil <<i <<' ' <<dataEvals.elem(i) <<' ' <<MSTS1(i) <<' ' <<MSTS2(i) <<endl;
  }

  if(verbose>0) LOG(0) <<"Minimal Spanning Tree size: " <<MSTS1(-1);

  // plots?
  if(verbose>1) plotData(nlp.bounds);

  if(verbose>2){ metrics().nearestHistograms(10, 20, 2.2); rai::wait(); }

  if(verbose>1){
    gnuplot("set size noratio; plot 'run.dat' us 1:3 w l");
    rai::wait();
  }

  return MSTS1(-1);
}

void Experiment::run(){
  NLP_Sampler_Options samopt;
  str baseDir = rai::getcwd_string();
  str path = STRING("ex_" <<opt.problem
                    <<'_' <<samopt.downhillMethod
                    <<'+' <<samopt.downhillNoiseMethod
                    <<'+' <<samopt.downhillRejectMethod
                    <<'_' <<samopt.interiorMethod
                    <<'_' <<samopt.downhillMaxSteps
                    <<'+' <<samopt.interiorBurnInSteps
                    <<'+' <<samopt.interiorSampleSteps
		    <<'_' <<samopt.seedMethod
		    <<samopt.seedCandidates);

  cout <<"===== " <<path <<" =====" <<endl;

  if(opt.outPath.N){
    path = opt.outPath;
    cout <<"===== " <<path <<" =====" <<endl;
  }

  rai::system(STRING("mkdir -p " <<path));
  chdir(path);

  double Dsum=0.;
  for(uint t=0;t<opt.runs;t++){
    rnd.seed(t);

    str prob = opt.problem;
    uint dim=0;
    int s = prob.find('.', false);
    if(s>0){ prob.getSubString(s+1,-1) >>dim; prob = prob.getSubString(0,s-1); }
    std::shared_ptr<Problem> P = loadProblem(prob, dim);

    if(opt.verbose>1){
      LOG(0) <<"=== problem dim: " <<P->nlp->getDimension() <<" bounds: " <<P->nlp->bounds;
    }

    Dsum += sample(*P->nlp, opt.verbose-1, 1.);
    rai::system(STRING("mv run.dat run." <<t <<".dat"));
    rai::system(STRING("mv samples.dat samples." <<t <<".dat"));
  }
  Dsum /= double(opt.runs);

  if(opt.verbose>0){
    uint mod = 1; //(opt.samples>100?10:1);
    double n = double(opt.samples)/mod;

    str pltcmd = "set size noratio\n plot ";
    pltcmd <<STRING(Dsum/pow(n, 1./1.)<<"*x**(1./1) lw 3, ");
    pltcmd <<STRING(Dsum/pow(n, 1./2.)<<"*x**(1./2) lw 3, ");
    //  pltcmd <<STRING(Dsum/pow(n, 1./3.)<<"*x**(1./3), ");
    for(uint t=0;t<opt.runs;t++){
      pltcmd <<STRING("'run." <<t <<".dat' us 1:3 w l not, ");
    }
    gnuplot(pltcmd);
    rai::wait();
  }

  chdir(baseDir);
}

