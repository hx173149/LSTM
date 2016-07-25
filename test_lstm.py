import numpy as np

#sigmoid2's function is equivelant np.tanh(x)
def sigmoid1(x):
  return 1. / (1 + np.exp(-x))
def sigmoid2(x):
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def devrivate_sigmoid1(x):
  return (np.exp(-x))/(1+np.exp(-x))**2
def devrivate_sigmoid2(x):
  return 1 - sigmoid2(x)**2

def loss(y,pred_y):
  return (pred_y-y)**2
def diff(y,pred_y):
  return y-pred_y

class LSTM:
  w_xi = np.random.rand(1,1)
  w_hi = np.random.rand(1,1)
  b_i = np.random.rand(1,1)
  w_xf = np.random.rand(1,1)
  w_hf = np.random.rand(1,1)
  b_f = np.random.rand(1,1)
  w_xo = np.random.rand(1,1)
  w_ho = np.random.rand(1,1)
  b_o = np.random.rand(1,1)
  w_xg = np.random.rand(1,1)
  w_hg = np.random.rand(1,1)
  b_g = np.random.rand(1,1)

  ht_last = 0.0
  ct_last = 0.0
  it = 0.0
  ft = 0.0
  ot = 0.0
  gt = 0.0
  ct = 0.0


  def ClearState(self):
    self.ht_last = 0.0
    self.ct_last = 0.0
    self.it = 0.0
    self.ft = 0.0
    self.ot = 0.0
    self.gt = 0.0
    self.ct = 0.0 


  def ForWard(self,x):
    self.it = sigmoid1(self.w_xi*x+self.w_hi*self.ht_last+self.b_i)
    self.ft = sigmoid1(self.w_xf*x+self.w_hf*self.ht_last+self.b_f)
    self.ot = sigmoid1(self.w_xo*x+self.w_ho*self.ht_last+self.b_o)
    self.gt = sigmoid2(self.w_xg*x+self.w_hg*self.ht_last+self.b_g)
    self.ct = self.ft*self.ct_last + self.it*self.gt
    return self.ot*sigmoid2(self.ct)
  
  def BackWard(self,x,lr,diff):
    delta_w_xi = lr*self.ot*devrivate_sigmoid2(self.ct)*self.gt*devrivate_sigmoid1(self.w_xi*x+self.w_hi*self.ht_last+self.b_i)*x*diff
    delta_w_hi = lr*self.ot*devrivate_sigmoid2(self.ct)*self.gt*devrivate_sigmoid1(self.w_xi*x+self.w_hi*self.ht_last+self.b_i)*self.ht_last*diff
    delta_b_i = lr*self.ot*devrivate_sigmoid2(self.ct)*self.gt*devrivate_sigmoid1(self.w_xi*x+self.w_hi*self.ht_last+self.b_i)*diff
    delta_w_xf = lr*self.ot*devrivate_sigmoid2(self.ct)*self.ct_last*devrivate_sigmoid1(self.w_xf*x+self.w_hf*self.ht_last+self.b_f)*x*diff
    delta_w_hf = lr*self.ot*devrivate_sigmoid2(self.ct)*self.ct_last*devrivate_sigmoid1(self.w_xf*x+self.w_hf*self.ht_last+self.b_f)*self.ht_last*diff
    delta_b_f = lr*self.ot*devrivate_sigmoid2(self.ct)*self.ct_last*devrivate_sigmoid1(self.w_xf*x+self.w_hf*self.ht_last+self.b_f)*diff
    delta_w_xo = lr*sigmoid2(self.ct)*devrivate_sigmoid1(self.w_xo*x+self.w_ho*self.ht_last+self.b_o)*x*diff
    delta_w_ho = lr*sigmoid2(self.ct)*devrivate_sigmoid1(self.w_xo*x+self.w_ho*self.ht_last+self.b_o)*self.ht_last*diff
    delta_b_o = lr*sigmoid2(self.ct)*devrivate_sigmoid1(self.w_xo*x+self.w_ho*self.ht_last+self.b_o)*diff
    delta_w_xg = lr*self.ot*devrivate_sigmoid2(self.ct)*self.it*devrivate_sigmoid2(self.w_xg*x+self.w_hg*self.ht_last+self.b_g)*x*diff
    delta_w_hg = lr*self.ot*devrivate_sigmoid2(self.ct)*self.it*devrivate_sigmoid2(self.w_xg*x+self.w_hg*self.ht_last+self.b_g)*self.ht_last*diff
    delta_b_g = lr*self.ot*devrivate_sigmoid2(self.ct)*self.it*devrivate_sigmoid2(self.w_xg*x+self.w_hg*self.ht_last+self.b_g)*diff
    return [delta_w_xi,delta_w_hi,delta_b_i,delta_w_xf,delta_w_hf,delta_b_f,delta_w_xo,delta_w_ho,delta_b_o,delta_w_xg,delta_w_hg,delta_b_g]
  
  def UpdateHtCt(self,):
    self.ht_last = self.ot*sigmoid2(self.ct)
    self.ct_last = self.ct
    return
  
  def UpdateWeight(self,delta):
    self.w_xi += delta[0]
    self.w_hi += delta[1]
    self.b_i += delta[2]
    self.w_xf += delta[3]
    self.w_hf += delta[4]
    self.b_f += delta[5]
    self.w_xo += delta[6]
    self.w_ho += delta[7]
    self.b_o += delta[8]
    self.w_xg += delta[9]
    self.w_hg += delta[10]
    self.b_g += delta[11]
    return

if __name__ == '__main__':
  x=[0.2,1.0,0.4,0.2]
  y=[0.2,0.9,0.8,0.4]
  max_iter = 5000
  test_lstm = LSTM()
  for step in range(0,max_iter):
    loss_sum = 0.0
    pred_y = []
    deltas=[]
    #every iterate must clean the ht_last and ct_last
    test_lstm.ClearState()
    print ('step: %d' % step)
    for t in range(0,4):
      #calculate forward, get the prediction
      pred = test_lstm.ForWard(x[t])
      #use the prediction calculate the backward and gradient
      deltas.append(test_lstm.BackWard(x[t],0.5,diff(y[t],pred)))
      #update ht_last and ct_last from t to t-1
      test_lstm.UpdateHtCt()
      #calculate the t time loss function
      loss_sum += loss(y[t],pred)[0][0]
      #save the t time prediction
      pred_y.append(pred[0][0])
    #use the gradients to update weights and bias parameters
    for i in range(0,4):
      test_lstm.UpdateWeight(deltas[i])
    #print steps state
    print ('loss:' + str(loss_sum))
    print ('predict_y: ' + str(pred_y))
