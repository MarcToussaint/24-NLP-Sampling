#include "problems.h"

#include <Kin/F_geometrics.h>
#include <Optim/utils.h>
#include <Optim/benchmarks.h>

//===========================================================================

BoxNLP::BoxNLP(uint dim){
  dimension = dim;
  featureTypes.resize(2*dimension);
  featureTypes = OT_ineq;
  bounds.resize(2, dimension);
  bounds[0] = -2.;
  bounds[1] = +2.;
  if(rai::getParameter<bool>("problemCosts")){
    featureTypes.append(rai::consts<ObjectiveType>(OT_sos,dimension));
  }
}

void BoxNLP::evaluate(arr& phi, arr& J, const arr& x){
  phi.resize(2*dimension);
  phi({0,dimension-1}) = -(x + 1.);
  phi({dimension,-1}) = x - 1.;

  J.resize(phi.N, x.N).setZero();
  for(uint i=0;i<dimension;i++){
    J(i,i) = -1.;
    J(dimension+i,i) = 1.;
  }

  if(featureTypes.N>2*dimension){
    arr d = x;
//    if(d(0)>0.) d(0) -= 1.; else d(0) += 1.;
    d -= ones(d.N);
    double w = 2.;
    phi.append(w*d);
    J.append(w*eye(dimension));
  }
}

//===========================================================================

ModesNLP::ModesNLP(uint dim, uint k){
  dimension = dim;
  cen = randn(k, dimension);
  cen.reshape(k, dimension);
  radii = .2 +10*rand(k);

#if 1
  k = 1+(uint(1)<<dimension);
  cen = randn(k, dimension);
  radii = consts(.1, k);
  cen[-1] = 0.;
  radii(-1) = .5;
  for(uint i=0;i<k-1;i++){
    for(uint d=0;d<dimension;d++){
      cen(i,d) = (i&(1<<d))? -1.: 1.;
    }
  }
#endif

  cen.reshape(-1, dimension);
  featureTypes.resize(1);
  featureTypes = OT_ineq;
  bounds.resize(2,dimension);
  bounds[0] = -1.2;
  bounds[1] = +1.2;
}

void ModesNLP::evaluate(arr& phi, arr& J, const arr& x){
  arr _x = x;
  _x.J_setId();

  uint k = cen.d0;
  arrA y(k);
  arr yval(k);
  for(uint i=0;i<k;i++){
    arr d = cen[i]-_x;
    double s = 1./(radii(i)*radii(i));
    y(i) = s * ~d*d - 1.;
    yval(i) = y(i).scalar();
  }

  phi = y(argmin(yval));
  J = phi.J_reset();
}

//===========================================================================

std::shared_ptr<Problem> loadProblem(str problem, uint dim){
  std::shared_ptr<Problem> P = make_shared<Problem>();

  if(problem == "box"){
    P->nlp = make_shared<BoxNLP>(dim);
  }
  else if(problem == "modes"){
    P->nlp = make_shared<ModesNLP>(dim);
  }
  if(problem == "tbox"){
    auto box = make_shared<BoxNLP>(dim);
    uint n = box->getDimension();
    arr A = eye(n);
    arr b = zeros(n);
    for(uint i=0;i<n;i++) A(i,i) = (box->bounds(1,i)-box->bounds(0,i))/2.;
    P->nlp = make_shared<NLP_LinTransformed>(box, A, b);
  }

  if(problem == "linear-program"){
    P->nlp = getBenchmarkFromCfg();
  }

  if(problem == "IK"){
    P->C.addFile(rai::raiPath("../rai-robotModels/scenarios/pandaSingle.g"));
    rai::Frame* f = P->C.addFrame("target", "table");
    f->setRelativePosition({.3, .2, .2});
    f->setShape(rai::ST_sphere, {.02}) .setColor({1., 1., 0.});

    P->komo.setConfig(P->C, false);
    P->komo.setTiming(1., 1, 1., 0);

    P->komo.addControlObjective({}, 0, 1e-1);
    P->komo.add_jointLimits();
    P->komo.addObjective({}, FS_positionDiff, {"l_gripper", "target"}, OT_eq, {1e1});

    P->nlp = P->komo.nlp();
  }

  if(problem == "cylinder-obstacle"){
    P->C.addFile("../cylinder-obstacle.g");
    P->manip = make_shared<ManipulationModelling>(P->C);

    P->manip->setup_inverse_kinematics(1e-1, false);

    //    manip.grasp_cylinder(1., "l_gripper", "cylinder", "l_palm");
    P->manip->komo->addObjective({}, FS_positionDiff, {"l_gripper", "dot"}, OT_eq, {1e1});

    for(uint coll=3;coll<=7;coll++){
      P->manip->no_collision({1.}, {STRING("l_panda_coll"<<coll), "obstacle"});
    }
    P->manip->no_collision({1.}, {"l_palm", "obstacle"});

    P->nlp = P->manip->komo->nlp();
  }

  if(problem == "torus-grasp"){
    P->C.addFile("../grasp-torus.g");

    rai::Frame* torus = P->C.addFrame("torus");
    torus->set_X()->rot.setRandom();

    P->komo.setConfig(P->C, false);
    P->komo.setTiming(1., 1, 1., 0);
    P->komo.addControlObjective({}, 0, 1e-1);
    //    P->komo.add_jointLimits();
    P->komo.addObjective({}, make_shared<F_TorusGraspEq>(.2, .02), {"l_gripper", "torus"}, OT_eq, {1e1});
    P->komo.addObjective({}, FS_negDistance, {"l_palm", "coll_torus"}, OT_ineq, {1e1});

    P->nlp = P->komo.nlp();
  }

  if(problem == "push"){
    P->C.addFile("../scene.g");
    rai::Joint *j = P->C["l_panda_finger_joint1"]->joint;
    j->setDofs(arr{.0});

    auto gripper = "l_gripper";
    auto palm = "l_palm";
    auto stick = "stick";
    auto obj = "box";
    auto table = "table";

    P->manip = make_shared<ManipulationModelling>(P->C, str{"push"}, StringA{"l_gripper"});

    P->manip->setup_sequence(4, 1e-1, 1e-1, false);
    P->manip->komo->addQuaternionNorms();

    //1,2: push
    P->manip->komo->addModeSwitch({1., 2.}, rai::SY_stable, {gripper, obj}, true);
    P->manip->komo->addModeSwitch({2., -1.}, rai::SY_stableOn, {table, obj}, false);
    P->manip->straight_push({1.,2.}, obj, gripper, table);
    P->manip->no_collision({2.}, {stick, obj}, .02);

    //3: pick
    P->manip->grasp_cylinder(3., gripper, stick, palm);
    P->manip->no_collision({3.}, {"l_panda_coll5", obj,
                              "l_panda_coll6", obj,
                              "l_panda_coll7", obj,
                              "l_palm", obj}, .02);

    //3,4: carry
    P->manip->komo->addModeSwitch({3., -1.}, rai::SY_stable, {gripper, stick}, true);

    //4: touch
    P->manip->komo->addObjective({4.}, FS_negDistance, {stick, "dot"}, OT_eq, {1e1});
    P->manip->no_collision({4.}, {stick, table,
                              palm, table}, .01);

    P->nlp = P->manip->komo->nlp();
  }

  CHECK(P->nlp, "");
  //  NLP_Viewer(nlp).display();  rai::wait();

  return P;
}

void testPush(){
  rai::Configuration C;
  C.addFile("../scene.g");

  rai::Joint *j = C["l_panda_finger_joint1"]->joint;
  j->setDofs(arr{.0});

  auto gripper = "l_gripper";
  auto palm = "l_palm";
  auto stick = "stick";
  auto obj = "box";
  auto table = "table";
//  auto qHome = C.getJointState();

//  C[obj]->setRelativePosition({-.0,.3-.055,.095});
//  C[obj]->setRelativeQuaternion({1.,0,0,0});

  for(uint i=0;i<20;i++){
//    arr qStart = C.getJointState();

    str info = STRING("push");
    ManipulationModelling seq(C, info, {"l_gripper"});

//    manip.setup_pick_and_place_waypoints(gripper, obj, 1e-1);

    seq.setup_sequence(4, 1e-1, 1e-1, false);
    seq.komo->addQuaternionNorms();

    //1,2: push

#if 0
    seq.komo->addModeSwitch({1., 2.}, rai::SY_stable, {gripper, obj}, true);
    seq.komo->addModeSwitch({2., -1.}, rai::SY_stable, {table, obj}, false);
#else
//    seq.komo->addFrameDof("obj_grasp", gripper, rai::JT_free, true, obj); //a permanent free stable gripper->grasp joint; and a snap grasp->object
//    seq.komo->addRigidSwitch(1., {"obj_grasp", obj});
    seq.komo->addFrameDof("obj_trans", table, rai::JT_transXY, false, obj); //a permanent moving(!) transXY joint table->trans, and a snap trans->obj
    seq.komo->addRigidSwitch(1., {"obj_trans", obj});

    seq.komo->addFrameDof("obj_place", "table", rai::JT_transXYPhi, true, obj); //a permanent stable joint table->placement, and a snap placement->hinge
    seq.komo->addRigidSwitch(2., {"obj_place", obj});
#endif

    seq.straight_push({1.,2.}, obj, gripper, table);
    seq.komo->addObjective({2.}, FS_poseRel, {gripper, obj}, OT_eq, {1e1}, {}, 1); //constant relative pose! (redundant for first switch option)
    seq.no_collision({2.}, {stick, obj}, .05);

    //3: pick
    seq.grasp_cylinder(3., gripper, stick, palm);
    seq.no_collision({3.}, {"l_panda_coll5", obj,
                              "l_panda_coll6", obj,
                              "l_panda_coll7", obj,
                              "l_palm", obj}, .02);

    //3,4: carry
    seq.komo->addModeSwitch({3., -1.}, rai::SY_stable, {gripper, stick}, true);

    //4: touch
    seq.komo->addObjective({4.}, FS_negDistance, {stick, "dot"}, OT_eq, {1e1});
    seq.no_collision({4.}, {stick, table,
                              palm, table}, .01);

    //random target position
//    manip.komo->addObjective({2.}, FS_position, {obj}, OT_eq, 1e1*arr{{2,3}, {1,0,0,0,1,0}}, .4*rand(3) - .2+arr{.0,.3,.0});
//    FILE("z.g") <<seq.komo->pathConfig; rai::wait();
    seq.solve(2);
    if(!seq.ret->feasible) continue;

    auto move0 = seq.sub_motion(0, 1e-1);
    move0->approachPush({.85, 1.}, gripper, .03);
    move0->no_collision({.0,.85}, {obj, "l_finger1",
                                   obj, "l_finger2",
                                   obj, "l_palm"}, .02);
    move0->no_collision({}, {table, "l_finger1",
                             table, "l_finger2"}, .0);
    move0->solve(2);
    if(!move0->ret->feasible) continue;

    auto move1 = seq.sub_motion(1, 1e-1);
//    move1->komo->addObjective({}, FS_positionRel, {gripper, "_push_start"}, OT_eq, 1e1*arr{{2,3},{1,0,0,0,0,1}});
//    move1->komo->addObjective({}, FS_negDistance, {gripper, obj}, OT_eq, {1e1}, {-.02});
    move1->komo->addObjective({}, FS_poseRel, {gripper, obj}, OT_eq, {1e1}, {}, 1); //constant relative pose! (redundant for first switch option)
    move1->solve(2);
    if(!move1->ret->feasible) continue;

    auto move2 = seq.sub_motion(2, 1e-1);
    move2->retractPush({.0, .15}, gripper, .03);
    move2->no_collision({.15}, {obj, "l_finger1",
                                obj, "l_finger2",
                                obj, "l_palm"}, .02);
    move2->solve(2);
    if(!move2->ret->feasible) continue;

    auto move3 = seq.sub_motion(3, 1e-1);
    move3->no_collision({}, {obj, stick,
                             table, stick}, .02);
    move3->solve(2);
    if(!move3->ret->feasible) continue;

    arr X = C.getFrameState();
    move0->play(C, 1.);
    C.attach(gripper, obj);
    move1->play(C, 1.);
    C.attach(table, obj);
    move2->play(C, 1.);
    C.attach(gripper, stick);
    move3->play(C, 1.);
    C.attach(table, stick);
    C.setFrameState(X);
  }
}
