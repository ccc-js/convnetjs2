var { assert } = require('./Util')
var Layers = require('./Layers')

// Net manages a set of layers
// For now constraints: Simple linear order of layers, first layer input last layer a cost layer
var Net = module.exports = function(options) {
  this.layers = [];
}

Net.prototype = {
  
  // takes a list of layer definitions and creates the network layer objects
  makeLayers: function(defs) {

    // few checks
    assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
    assert(defs[0].type === 'input', 'Error! First layer must be the input layer, to declare size of inputs');

    // desugar layer_defs for adding activation, dropout layers etc
    var desugar = function() { // de-sugar: 去糖化，加入一些使用者不須操心的預設層次 ...
      var new_defs = [];
      for(var i=0;i<defs.length;i++) {
        var def = defs[i];
        
        if(def.type==='softmax' || def.type==='svm') { // softmax, svm 前先加入一層全連接層
          // add an fc layer here, there is no reason the user should
          // have to worry about this and we almost always want to
          new_defs.push({type:'fc', num_neurons: def.num_classes}); 
        }

        if(def.type==='regression') { // regression 前先加入一層全連接層
          // add an fc layer here, there is no reason the user should
          // have to worry about this and we almost always want to
          new_defs.push({type:'fc', num_neurons: def.num_neurons});
        }

        if((def.type==='fc' || def.type==='conv') 
            && typeof(def.bias_pref) === 'undefined'){
          def.bias_pref = 0.0;
          if(typeof def.activation !== 'undefined' && def.activation === 'relu') {
            // 使用 LeakyRelu 避免 Relu 落入停止不動點 
            def.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
            // otherwise it's technically possible that a relu unit will never turn on (by chance)
            // and will never get any gradient and never contribute any computation. Dead relu.
          }
        }

        new_defs.push(def);

        if(typeof def.activation !== 'undefined') {
          if(def.activation==='relu') { new_defs.push({type:'relu'}); }
          else if (def.activation==='sigmoid') { new_defs.push({type:'sigmoid'}); }
          else if (def.activation==='tanh') { new_defs.push({type:'tanh'}); }
          else if (def.activation==='maxout') {
            // create maxout activation, and pass along group size, if provided
            var gs = def.group_size !== 'undefined' ? def.group_size : 2;
            new_defs.push({type:'maxout', group_size:gs});
          }
          else { console.log('ERROR unsupported activation ' + def.activation); }
        }
        if(typeof def.drop_prob !== 'undefined' && def.type !== 'dropout') { // 有設定 drop_prob，所以要執行 dropout 捏掉一些節點
          new_defs.push({type:'dropout', drop_prob: def.drop_prob});
        }

      }
      return new_defs;
    }
    defs = desugar(defs);

    // create the layers
    this.layers = [];
    for(var i=0;i<defs.length;i++) {
      var def = defs[i];
      if(i>0) {
        var prev = this.layers[i-1];
        def.in_sx = prev.out_sx;
        def.in_sy = prev.out_sy;
        def.in_depth = prev.out_depth;
      }

      switch(def.type) {
        case 'fc': this.layers.push(new Layers.FullyConnLayer(def)); break;
        case 'lrn': this.layers.push(new Layers.LocalResponseNormalizationLayer(def)); break;
        case 'dropout': this.layers.push(new Layers.DropoutLayer(def)); break;
        case 'input': this.layers.push(new Layers.InputLayer(def)); break;
        case 'softmax': this.layers.push(new Layers.SoftmaxLayer(def)); break;
        case 'regression': this.layers.push(new Layers.RegressionLayer(def)); break;
        case 'conv': this.layers.push(new Layers.ConvLayer(def)); break;
        case 'pool': this.layers.push(new Layers.PoolLayer(def)); break;
        case 'relu': this.layers.push(new Layers.ReluLayer(def)); break;
        case 'sigmoid': this.layers.push(new Layers.SigmoidLayer(def)); break;
        case 'tanh': this.layers.push(new Layers.TanhLayer(def)); break;
        case 'maxout': this.layers.push(new Layers.MaxoutLayer(def)); break;
        case 'svm': this.layers.push(new Layers.SVMLayer(def)); break;
        default: console.log('ERROR: UNRECOGNIZED LAYER TYPE: ' + def.type);
      }
    }
  },

  // forward prop the network. 
  // The trainer class passes is_training = true, but when this function is
  // called from outside (not from the trainer), it defaults to prediction mode
  forward: function(V, is_training) {
    if(typeof(is_training) === 'undefined') is_training = false;
    var act = this.layers[0].forward(V, is_training);
    // 從輸入層開始逐層正向傳遞
    for(var i=1;i<this.layers.length;i++) {
      act = this.layers[i].forward(act, is_training);
    }
    return act;
  },

  getCostLoss: function(V, y) {
    this.forward(V, false);
    var N = this.layers.length;
    var loss = this.layers[N-1].backward(y);
    return loss;
  },
  
  // backprop: compute gradients wrt all parameters
  backward: function(y) {
    var N = this.layers.length;
    // 從輸出層開始逐層反向傳遞
    var loss = this.layers[N-1].backward(y); // last layer assumed to be loss layer
    for(var i=N-2;i>=0;i--) { // first layer assumed input
      this.layers[i].backward();
    }
    return loss; // loss 代表損失值，通常是錯誤率
  },
  getParamsAndGrads: function() {
    // accumulate parameters and gradients for the entire network
    var response = [];
    for(var i=0;i<this.layers.length;i++) {
      var layer_reponse = this.layers[i].getParamsAndGrads();
      for(var j=0;j<layer_reponse.length;j++) {
        response.push(layer_reponse[j]);
      }
    }
    return response;
  },
  getPrediction: function() {
    // this is a convenience function for returning the argmax
    // prediction, assuming the last layer of the net is a softmax
    var S = this.layers[this.layers.length-1];
    assert(S.layer_type === 'softmax', 'getPrediction function assumes softmax as last layer of the net!');

    var p = S.out_act.w;
    var maxv = p[0];
    var maxi = 0;
    for(var i=1;i<p.length;i++) {
      if(p[i] > maxv) { maxv = p[i]; maxi = i;}
    }
    return maxi; // return index of the class with highest class probability
  },
  toJSON: function() {
    var json = {};
    json.layers = [];
    for(var i=0;i<this.layers.length;i++) {
      json.layers.push(this.layers[i].toJSON());
    }
    return json;
  },
  fromJSON: function(json) {
    this.layers = [];
    for(var i=0;i<json.layers.length;i++) {
      var Lj = json.layers[i]
      var t = Lj.layer_type;
      var L;
      if(t==='input') { L = new Layers.InputLayer(); }
      if(t==='relu') { L = new Layers.ReluLayer(); }
      if(t==='sigmoid') { L = new Layers.SigmoidLayer(); }
      if(t==='tanh') { L = new Layers.TanhLayer(); }
      if(t==='dropout') { L = new Layers.DropoutLayer(); }
      if(t==='conv') { L = new Layers.ConvLayer(); }
      if(t==='pool') { L = new Layers.PoolLayer(); }
      if(t==='lrn') { L = new Layers.LocalResponseNormalizationLayer(); }
      if(t==='softmax') { L = new Layers.SoftmaxLayer(); }
      if(t==='regression') { L = new Layers.RegressionLayer(); }
      if(t==='fc') { L = new Layers.FullyConnLayer(); }
      if(t==='maxout') { L = new Layers.MaxoutLayer(); }
      if(t==='svm') { L = new Layers.SVMLayer(); }
      L.fromJSON(Lj);
      this.layers.push(L);
    }
  }
}