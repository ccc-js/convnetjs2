
# ConvNetJs2

ConvNetJs2 is a successor of the [ConvNetJS project](https://github.com/karpathy/convnetjs) 。

ConvNetJs2 is a Javascript implementation of Neural networks, together with nice browser-based demos. It currently supports:

- Common **Neural Network modules** (fully connected layers, non-linearities)
- Classification (SVM/Softmax) and Regression (L2) **cost functions**
- Ability to specify and train **Convolutional Networks** that process images
- An experimental **Reinforcement Learning** module, based on Deep Q Learning

For much more information, see the ConvNetJs2 main page at [https://ccc-js.github.io/convnetjs2demo/](https://ccc-js.github.io/convnetjs2demo/)

**Note**: I am not actively maintaining ConvNetJS anymore because I simply don't have time. I think the npm repo might not work at this point.

## Online Demos
- [Convolutional Neural Network on MNIST digits](https://ccc-js.github.io/convnetjs2demo/demo/mnist.html)
- [Convolutional Neural Network on CIFAR-10](https://ccc-js.github.io/convnetjs2demo/demo/cifar10.html)
- [Toy 2D data](https://ccc-js.github.io/convnetjs2demo/demo/classify2d.html)
- [Toy 1D regression](https://ccc-js.github.io/convnetjs2demo/demo/regression.html)
- [Training an Autoencoder on MNIST digits](https://ccc-js.github.io/convnetjs2demo/demo/autoencoder.html)
- [Deep Q Learning Reinforcement Learning demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
- [Image Regression ("Painting")](https://ccc-js.github.io/convnetjs2demo/demo/image_regression.html)
- [Comparison of SGD/Adagrad/Adadelta on MNIST](https://ccc-js.github.io/convnetjs2demo/demo/trainers.html)

## Example Code

You may found more node.js examples for the ConvNetJs2 in the following project.

* https://github.com/ccc-js/convnetjs2example

Here's a minimum example of defining a **2-layer neural network** and training
it on a single data point:

```javascript
// species a 2-layer neural network with one hidden layer of 20 neurons
var layer_defs = [];
// input layer declares size of input. here: 2-D data
// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
// then the first two dimensions (sx, sy) will always be kept at size 1
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'}); 
// declare the linear classifier on top of the previous hidden layer
layer_defs.push({type:'softmax', num_classes:10});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// forward a random data point through the network
var x = new convnetjs.Vol([0.3, -0.5]);
var prob = net.forward(x); 

// prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
console.log('probability that x is class 0: ' + prob.w[0]); // prints 0.50101

var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
trainer.train(x, 0); // train the network, specifying that x is class zero

var prob2 = net.forward(x);
console.log('probability that x is class 0: ' + prob2.w[0]);
// now prints 0.50374, slightly higher than previous 0.50101: the networks
// weights have been adjusted by the Trainer to give a higher probability to
// the class we trained the network with (zero)
```

and here is a small **Convolutional Neural Network** if you wish to predict on images:

```javascript
var layer_defs = [];
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3}); // declare size of input
// output Vol is of size 32x32x3 here
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
// the layer will perform convolution with 16 kernels, each of size 5x5.
// the input will be padded with 2 pixels on all sides to make the output Vol of the same size
// output Vol will thus be 32x32x16 at this point
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 16x16x16 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 16x16x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 4x4x20 here
layer_defs.push({type:'softmax', num_classes:10});
// output Vol is of size 1x1x10 here

net = new convnetjs.Net();
net.makeLayers(layer_defs);

// helpful utility for converting images into Vols is included
var x = convnetjs.img_to_vol(document.getElementById('some_image'))
var output_probabilities_vol = net.forward(x)
```

## Getting Started
A [Getting Started](https://ccc-js.github.io/convnetjs2demo/started.html) tutorial is available on main page.

The full [Documentation](https://ccc-js.github.io/convnetjs2demo/docs.html) can also be found there.

See the **releases** page for this project to get the minified, compiled library, and a direct link to is also available below for convenience (but please host your own copy)

- [convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/build/convnet.js)

## Compiling the library from src/ to build/
If you would like to add features to the library, you will have to change the code in `src/` and then compile the library into the `build/` directory. The compilation script simply concatenates files in `src/` and then minifies the result.

The compilation is done using an npm script by browserify

    $ npm run build

The output files will be in `build/`

## Use in Node

The library is also available on *node.js*:

1. Install it: `$ npm install convnetjs2`
2. Use it: `var convnetjs = require("convnetjs2");`

## License

MIT
