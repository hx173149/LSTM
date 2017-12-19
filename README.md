# A simple code for implementate the LSTM

---

**Feed forward propagation:**

![equation](http://latex.codecogs.com/gif.latex?%5Csigma%20%5Cleft%20%28%20x%20%5Cright%20%29%3D%20%5Cleft%20%28%201&plus;e%5E%7B-x%7D%20%5Cright%20%29%5E%7B-1%7D)

![equation](http://latex.codecogs.com/gif.latex?%5Cphi%5Cleft%20%28%20x%20%5Cright%20%29%3D%20%5Cfrac%7Be%5E%7Bx%7D-e%5E%7B-x%7D%7D%7Be%5E%7Bx%7D&plus;e%5E%7B-x%7D%7D)

![equation](http://latex.codecogs.com/gif.latex?i_%7Bt%7D%3D%20%5Csigma%20%5Cleft%20%28W_%7Bxi%7Dx_%7Bt%7D&plus;W_%7Bhi%7Dh_%7Bt-1%7D&plus;b_%7Bi%7D%5Cright%20%29)

![equation](http://latex.codecogs.com/gif.latex?f_%7Bt%7D%3D%20%5Csigma%20%5Cleft%20%28W_%7Bxf%7Dx_%7Bt%7D&plus;W_%7Bhf%7Dh_%7Bt-1%7D&plus;b_%7Bf%7D%5Cright%20%29)

![equation](http://latex.codecogs.com/gif.latex?o_%7Bt%7D%3D%20%5Csigma%20%5Cleft%20%28W_%7Bxo%7Dx_%7Bt%7D&plus;W_%7Bho%7Dh_%7Bt-1%7D&plus;b_%7Bo%7D%5Cright%20%29)

![equation](http://latex.codecogs.com/gif.latex?g_%7Bt%7D%3D%20%5Cphi%20%5Cleft%20%28W_%7Bxg%7Dx_%7Bt%7D&plus;W_%7Bhg%7Dh_%7Bt-1%7D&plus;b_%7Bg%7D%5Cright%20%29)

![equation](http://latex.codecogs.com/gif.latex?c_%7Bt%7D%20%3D%20f_%7Bt%7D%20%5Codot%20c_%7Bt-1%7D&plus;i_%7Bt%7D%20%5Codot%20g_%7Bt%7D)

![equation](http://latex.codecogs.com/gif.latex?h_%7Bt%7D%20%3D%20o_%7Bt%7D%20%5Codot%20%5Cphi%5Cleft%20%28c_%7Bt%7D%5Cright%20%29)

hello world
