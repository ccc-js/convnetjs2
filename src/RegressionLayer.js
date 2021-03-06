const Util = require('./Util')
// RegressionLayer 是最小平方法的層
// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.
var RegressionLayer = module.exports = function(opt) {
  var opt = opt || {};

  // computed
  this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
  this.out_depth = this.num_inputs;
  this.out_sx = 1;
  this.out_sy = 1;
  this.layer_type = 'regression';
}

RegressionLayer.prototype = {
  // 這層是輸出層，不用再向前傳遞了
  forward: function(V, is_training) {
    this.in_act = V;
    this.out_act = V;
    return V; // identity function
  },
  // y is a list here of size num_inputs
  // or it can be a number if only one value is regressed
  // or it can be a struct {dim: i, val: x} where we only want to 
  // regress on dimension i and asking it to have value x
  backward: function(y) { // y 是正確輸出值

    // compute and accumulate gradient wrt weights and bias of this layer
    var x = this.in_act;
    x.dw = Util.zeros(x.w.length); // zero out the gradient of input Vol
    var loss = 0.0;
    if(y instanceof Array || y instanceof Float64Array) { // 輸出是陣列的情況
      for(var i=0;i<this.out_depth;i++) {
        var dy = x.w[i] - y[i]; // 計算網路輸出 x.w[i] 與正確輸出 y[i] 之間的誤差
        x.dw[i] = dy;
        loss += 0.5*dy*dy;
      }
    } else if(typeof y === 'number') { // 輸出是單一數值的情況
      // lets hope that only one number is being regressed
      var dy = x.w[0] - y;
      x.dw[0] = dy;
      loss += 0.5*dy*dy;
    } else { // 否則輸出 y 應該是 {dim:..., val:...} 的結構
      // assume it is a struct with entries .dim and .val
      // and we pass gradient only along dimension dim to be equal to val
      var i = y.dim;
      var yi = y.val;
      var dy = x.w[i] - yi;
      x.dw[i] = dy;
      loss += 0.5*dy*dy;
    }
    return loss;
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
    json.num_inputs = this.num_inputs;
    return json;
  },
  fromJSON: function(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
    this.num_inputs = json.num_inputs;
  }
}