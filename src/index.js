var convnetjs = module.exports = require('./Util')
Object.assign(convnetjs, 
{
  ConvLayer: require('./ConvLayer'),
  DropoutLayer: require('./DropoutLayer'),
  FullyConnLayer: require('./FullyConnLayer'),
  InputLayer: require('./InputLayer'),
  LocalResponseNormalizationLayer: require('./LocalResponseNormalizationLayer'),
  MagicNet: require('./MagicNet'),
  MaxoutLayer: require('./MaxoutLayer'),
  Net: require('./Net'),
  PoolLayer: require('./PoolLayer'),
  SigmoidLayer: require('./SigmoidLayer'),
  SoftmaxLayer: require('./SoftmaxLayer'),
  SVMLayer: require('./SVMLayer'),
  TanhLayer: require('./TanhLayer'),
  Trainer: require('./Trainer'),
  Util: require('./Util'),
  Vol: require('./Vol'),
  VolUtil: require('./VolUtil')
})

convnetjs.SGDTrainer = convnetjs.Trainer // backwards compatibility
convnetjs.augment = convnetjs.VolUtil.augment
convnetjs.img_to_vol = convnetjs.VolUtil.img_to_vol
