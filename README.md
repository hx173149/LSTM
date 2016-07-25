# A simple code for implementate the LSTM

---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

**Feed forward propagation:**

$$\sigma \left ( x \right )= \left ( 1+e^{-x} \right )^{-1}$$
$$\phi\left ( x \right )= \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$
$$i_{t}= \sigma \left (W_{xi}x_{t}+W_{hi}h_{t-1}+b_{i}\right )$$
$$f_{t}= \sigma \left (W_{xf}x_{t}+W_{hf}h_{t-1}+b_{f}\right )$$
$$o_{t}= \sigma \left (W_{xo}x_{t}+W_{ho}h_{t-1}+b_{o}\right )$$
$$g_{t}= \phi \left (W_{xg}x_{t}+W_{hg}h_{t-1}+b_{g}\right )$$
$$c_{t} = f_{t} \odot c_{t-1}+i_{t} \odot g_{t}$$
$$h_{t} = o_{t} \odot \phi\left (c_{t}\right )$$


