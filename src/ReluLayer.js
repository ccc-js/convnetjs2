const Util = require('./Util')

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
var ReluLayer = module.exports = function(opt) {
  var opt = opt || {};

  // computed
  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.layer_type = 'relu';
}
ReluLayer.prototype = {
  forward: function(V, is_training) {
    this.in_act = V;
    var V2 = V.clone();
    var N = V.w.length;
    var V2w = V2.w;
    for(var i=0;i<N;i++) { 
      if(V2w[i] < 0) V2w[i] = 0; // threshold at 0  (Relu 就是把低於零的截掉)
    }
    this.out_act = V2;
    return this.out_act;
  },
  backward: function() {
    var V = this.in_act; // we need to set dw of this
    var V2 = this.out_act;
    var N = V.w.length;
    V.dw = Util.zeros(N); // zero out gradient wrt data
    for(var i=0;i<N;i++) {
      if(V2.w[i] <= 0) V.dw[i] = 0; // threshold (低於零的梯度為 0)
      else V.dw[i] = V2.dw[i]; // (高於零的梯度不變)
    }
  },
  getParamsAndGrads: function() {
    return [];
  },
  toJSON: function() {
    var json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    return json;
  },
  fromJSON: function(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
  }
}