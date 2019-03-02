const Vol = require('./Vol')
const Util = require('./Util')

var FullyConnLayer = module.exports = function(opt) {
  var opt = opt || {};

  // required
  // ok fine we will allow 'filters' as the word as well
  this.out_depth = typeof opt.num_neurons !== 'undefined' ? opt.num_neurons : opt.filters;

  // optional 
  this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
  this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

  // computed
  this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
  this.out_sx = 1;
  this.out_sy = 1;
  this.layer_type = 'fc';

  // initializations
  var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
  this.filters = [];
  for(var i=0;i<this.out_depth ;i++) { this.filters.push(new Vol(1, 1, this.num_inputs)); }
  this.biases = new Vol(1, 1, this.out_depth, bias);
}

FullyConnLayer.prototype = {
  forward: function(V, is_training) {
    this.in_act = V;
    var A = new Vol(1, 1, this.out_depth, 0.0);
    var Vw = V.w;
    for(var i=0;i<this.out_depth;i++) {
      var a = 0.0;
      var wi = this.filters[i].w;
      for(var d=0;d<this.num_inputs;d++) {
        a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
      }
      a += this.biases.w[i];
      A.w[i] = a;
    }
    this.out_act = A;
    return this.out_act;
  },
  backward: function() {
    var V = this.in_act;
    V.dw = Util.zeros(V.w.length); // zero out the gradient in input Vol
    
    // compute gradient wrt weights and data
    for(var i=0;i<this.out_depth;i++) {
      var tfi = this.filters[i];
      var chain_grad = this.out_act.dw[i];
      for(var d=0;d<this.num_inputs;d++) {
        V.dw[d] += tfi.w[d]*chain_grad; // grad wrt input data
        tfi.dw[d] += V.w[d]*chain_grad; // grad wrt params
      }
      this.biases.dw[i] += chain_grad;
    }
  },
  getParamsAndGrads: function() {
    var response = [];
    for(var i=0;i<this.out_depth;i++) {
      response.push({params: this.filters[i].w, grads: this.filters[i].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
    }
    response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
    return response;
  },
  toJSON: function() {
    var json = {};
    json.out_depth = this.out_depth;
    json.out_sx = this.out_sx;
    json.out_sy = this.out_sy;
    json.layer_type = this.layer_type;
    json.num_inputs = this.num_inputs;
    json.l1_decay_mul = this.l1_decay_mul;
    json.l2_decay_mul = this.l2_decay_mul;
    json.filters = [];
    for(var i=0;i<this.filters.length;i++) {
      json.filters.push(this.filters[i].toJSON());
    }
    json.biases = this.biases.toJSON();
    return json;
  },
  fromJSON: function(json) {
    this.out_depth = json.out_depth;
    this.out_sx = json.out_sx;
    this.out_sy = json.out_sy;
    this.layer_type = json.layer_type;
    this.num_inputs = json.num_inputs;
    this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
    this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
    this.filters = [];
    for(var i=0;i<json.filters.length;i++) {
      var v = new Vol(0,0,0,0);
      v.fromJSON(json.filters[i]);
      this.filters.push(v);
    }
    this.biases = new Vol(0,0,0,0);
    this.biases.fromJSON(json.biases);
  }
}