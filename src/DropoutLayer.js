const Util = require('./Util')

// An inefficient dropout layer
// Note this is not most efficient implementation since the layer before
// computed all these activations and now we're just going to drop them :(
// same goes for backward pass. Also, if we wanted to be efficient at test time
// we could equivalently be clever and upscale during train and copy pointers during test
// todo: make more efficient.
var DropoutLayer = module.exports = function(opt) {
  var opt = opt || {};

  // computed
  this.out_sx = opt.in_sx;
  this.out_sy = opt.in_sy;
  this.out_depth = opt.in_depth;
  this.layer_type = 'dropout';
  this.drop_prob = typeof opt.drop_prob !== 'undefined' ? opt.drop_prob : 0.5;
  this.dropped = Util.zeros(this.out_sx*this.out_sy*this.out_depth);
}
DropoutLayer.prototype = {
  forward: function(V, is_training) {
    this.in_act = V;
    if(typeof(is_training)==='undefined') { is_training = false; } // default is prediction mode
    var V2 = V.clone();
    var N = V.w.length;
    if(is_training) { // 在訓練階段，隨機的掐掉一些權重。
      // do dropout
      for(var i=0;i<N;i++) {
        if(Math.random()<this.drop_prob) { V2.w[i]=0; this.dropped[i] = true; } // drop!  這個節點已經被掐掉了 ..
        else {this.dropped[i] = false;}
      }
    } else { // 在使用階段，將權重 * drop_prob 以得到正規化後的結果。
      // scale the activations during prediction
      for(var i=0;i<N;i++) { V2.w[i]*=this.drop_prob; }
    }
    this.out_act = V2;
    return this.out_act; // dummy identity function for now
  },
  backward: function() {
    var V = this.in_act; // we need to set dw of this
    var chain_grad = this.out_act;
    var N = V.w.length;
    V.dw = Util.zeros(N); // zero out gradient wrt data
    for(var i=0;i<N;i++) {
      if(!(this.dropped[i])) { // 反向傳遞: 單純將梯度傳回去。
        V.dw[i] = chain_grad.dw[i]; // copy over the gradient
      }
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
    json.drop_prob = this.drop_prob;
    return json;
  },
  fromJSON: function(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type; 
    this.drop_prob = json.drop_prob;
  }
}