Include: <../rai-robotModels/scenarios/pandaSingle.g>

box(table): { joint: rigid, Q:[.4, .0, .25], shape: ssBox, size: [.15, .4, .4, .005], contact: 1, mass: .1 }

dot(table): { Q: [-.6, .6, .1], shape:sphere, size:[.02], color: [1., 1., .5] }

stick (table): {  joint: rigid, shape: capsule, Q: "t(.75 .0 .1) d(90 1 0 0) ", size: [.6, .02], color: [.6], contact: 1
}

Edit panda_collCameraWrist: { shape: marker, contact: 0 }
