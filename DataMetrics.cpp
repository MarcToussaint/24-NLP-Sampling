#include "DataMetrics.h"
#include <Algo/minimalSpanningTree.h>
#include <Gui/plot.h>

#ifdef RAI_OPENCV
#  include <Perception/opencv.h>
#  include <opencv2/imgproc.hpp>
#endif

arr DataMetrics::sequentialDistance(){
  ANN seq;

  arr dis;

  uintA _idx;
  arr _sqrDists;
  double total=0.;

  for(uint i=0;i<D.d0;i++){
    arr x = D[i];

    if(seq.X.d0){
      seq.getkNN(_sqrDists, _idx, x, 1);
      double d = sqrt(_sqrDists(0));
      total += d;
      dis.append(total); // / pow(double(i), 1./double(D.d1)));
    }

    seq.append(x);
  }

  return dis;
}

arr DataMetrics::sequentialMinimalSpanningTree(double coeff){
  ANN ann;

  arr MST_sizes;
  MST_sizes = arr{0.};

  uint k=10;
  uintA _idx;
  arr _sqrDists;

  uint mod = 1;// (D.d0>100?10:1);

  rai::Array<DoubleEdge> edges;

  for(uint i=0;i<D.d0;i++){

    arr x = D[i];

    CHECK_EQ(i, ann.X.d0, "");
    if(i){
      ann.getkNN(_sqrDists, _idx, x, (i<k?i:k));
      for(uint j=0;j<_idx.N;j++){
        double w = _sqrDists(j);
        if(coeff==1.) w = sqrt(w);
        else if(coeff==2.) {}
        else w = pow(w, 0.5* coeff);
        edges.append(DoubleEdge{i, _idx(j), w});
      }

      if(!(i%mod)){
        auto minSpanTree = minimalSpanningTree(i+1, edges);
        MST_sizes.append(get<0>(minSpanTree)); // / pow(double(i), 1./double(D.d1)));

#if 0
        plot()->Clear();
        plot()->Points(D({0,i}));
        for(uint e: get<1>(minSpanTree)){
          plot()->Line((D[edges(e).i], D[edges(e).j]).reshape(2,2));
        }
        plot()->update(true, STRING("cost: " <<get<0>(minSpanTree)));
#endif
      }
    }

    ann.append(x);
  }

  return MST_sizes;

}

arr DataMetrics::nearestHistograms(uint kNearest, uint bins, double maxDist){
  arr dists;
  uintA idx;

  ANN ann;
  ann.setX(D);

  double del = double(maxDist)/double(bins);

  arr sumDist = zeros(kNearest+1);
  arr sqrDist = zeros(kNearest+1);

  arr hist = zeros(kNearest, bins);
  for(uint i=0;i<D.d0;i++){
    arr x = D[i];

    ann.getkNN(dists, idx, x, kNearest+1);

    for(double& d:dists) d = sqrt(d);

    sumDist += dists;
    sqrDist += ::sqr(dists);

    for(uint k=0;k<kNearest;k++){
      uint bin = dists(k+1)/del;
      if(bin>=bins) bin=bins-1;
      hist(k, bin) += 1.;
    }
  }
  hist /= D.d0;
  sumDist /= D.d0;
  sqrDist /= D.d0;
  sqrDist -= sqr(sumDist);
  sqrDist = sqrt(sqrDist);

  std::cout <<"average distances: " <<sumDist <<std::endl;
  std::cout <<"svd     distances: " <<sqrDist <<std::endl;

  FILE("z.hist") <<(~hist).modRaw() <<std::endl;
  rai::String plt;
  plt <<"plot 'z.hist' us ($0*" <<del <<"):1 t 'near1'";
  for(uint k=1;k<kNearest;k++) plt <<", '' us ($0*" <<del <<"):"<<k+1 <<" t 'near" <<k+1 <<"'";
  //    std::cout <<plt <<std::endl;
  gnuplot(plt);

  return hist;
}

#ifdef RAI_OPENCV
double DataMetrics::EarthMoverDistance(const arr& Dref){
  floatA A = rai::convert<float>(Dref), B = rai::convert<float>(D);
  CHECK_EQ(A.d0, B.d0, "");
  CHECK_EQ(A.d1, B.d1, "");
  A.insColumns(0);
  B.insColumns(0);
  for(uint i=0;i<A.d0;i++) A(i,0) = 1.f;
  for(uint i=0;i<B.d0;i++) B(i,0) = 1.f;

  float d = cv::EMD(CV(A), CV(B), cv::DIST_L2);
  return d;
}
#else
double DataMetrics::EarthMoverDistance(const arr& Dref){ NICO }
#endif

arr histogram(const arr& data, const arr& minMax, uint bins){
  arr H = zeros(bins);
  //  double m = min(data), M = max(data);
  double m = minMax(0), M = minMax(1);
  double del = (M-m)/double(bins);

  for(uint i=0;i<data.N;i++){
    int bin = (data.elem(i)-m)/del;
    if(bin<0) continue; //bin=0.;
    else if(bin>=(int)H.N) continue; //bin=H.N-1;
    H(bin) += 1.;
  }
  H /= sum(H);
  return H;
}
