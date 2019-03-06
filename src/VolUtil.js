const Util = require('./Util')
const Vol = require('./Vol')

var U = module.exports = {}

/* 擴增，調整： 例如在 images-demo.js 裡有

        test_variation = convnetjs.augment(test_variation, image_dimension, dx, dy, false);
      }
      
      if(random_flip){
        test_variation = convnetjs.augment(test_variation, image_dimension, 0, 0, Math.random()<0.5); 
      }

在 mnistPredict.js 裡有

      x = convnetjs.augment(x, 24)
*/
// Volume utilities
// intended for use with data augmentation (資料擴增)
// crop is the size of output (調整後大小)
// dx,dy are offset wrt incoming volume, of the shift (位移量 ?? 應該是為了讓訓練效果在位移後不變，能夠抓到真正的特徵)
// fliplr is boolean on whether we also want to flip left<->right (是否要左右翻轉)
U.augment = function(V, crop, dx, dy, fliplr) {
  // note assumes square outputs of size crop x crop
  if(typeof(fliplr)==='undefined') var fliplr = false;
  if(typeof(dx)==='undefined') var dx = Util.randi(0, V.sx - crop);
  if(typeof(dy)==='undefined') var dy = Util.randi(0, V.sy - crop);
  
  // randomly sample a crop in the input volume
  var W;
  if(crop !== V.sx || dx!==0 || dy!==0) {
    W = new Vol(crop, crop, V.depth, 0.0);
    for(var x=0;x<crop;x++) {
      for(var y=0;y<crop;y++) {
        if(x+dx<0 || x+dx>=V.sx || y+dy<0 || y+dy>=V.sy) continue; // oob
        for(var d=0;d<V.depth;d++) {
          W.set(x,y,d,V.get(x+dx,y+dy,d)); // copy data over
        }
      }
    }
  } else {
    W = V;
  }

  if(fliplr) {
    // flip volume horziontally
    var W2 = W.cloneAndZero();
    for(var x=0;x<W.sx;x++) {
      for(var y=0;y<W.sy;y++) {
        for(var d=0;d<W.depth;d++) {
          W2.set(x,y,d,W.get(W.sx - x - 1,y,d)); // copy data over
        }
      }
    }
    W = W2; //swap
  }
  return W;
}

// 將 HTML DOM 中的影像 Image 轉換為 Vol 物件
// img is a DOM element that contains a loaded image
// returns a Vol of size (W, H, 4). 4 is for RGBA
U.img_to_vol = function(img, convert_grayscale) {

  if(typeof(convert_grayscale)==='undefined') var convert_grayscale = false;

  var canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  var ctx = canvas.getContext("2d");

  // due to a Firefox bug
  try {
    ctx.drawImage(img, 0, 0);
  } catch (e) {
    if (e.name === "NS_ERROR_NOT_AVAILABLE") {
      // sometimes happens, lets just abort
      return false;
    } else {
      throw e;
    }
  }

  try {
    var img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  } catch (e) {
    if(e.name === 'IndexSizeError') {
      return false; // not sure what causes this sometimes but okay abort
    } else {
      throw e;
    }
  }

  // prepare the input: get pixels and normalize them
  var p = img_data.data;
  var W = img.width;
  var H = img.height;
  var pv = []
  for(var i=0;i<p.length;i++) {
    pv.push(p[i]/255.0-0.5); // normalize image pixels to [-0.5, 0.5]
  }
  var x = new Vol(W, H, 4, 0.0); //input volume (image)
  x.w = pv;

  if(convert_grayscale) {
    // flatten into depth=1 array
    var x1 = new Vol(W, H, 1, 0.0);
    for(var i=0;i<W;i++) {
      for(var j=0;j<H;j++) {
        x1.set(i,j,0,x.get(i,j,0));
      }
    }
    x = x1;
  }

  return x;
}