from typing import Type
import numpy as np
import pprint as pp # 🤨 *vine boom*
import random
import matplotlib.pyplot as plt


class Layer:
	def __init__(self, nIn, nOut, activ = "relu"):
		self.weights = np.random.uniform(-1, 1, size=(nIn, nOut));
		self.biases  = np.random.uniform(-1, 1, size= nOut      );
		self.costw   = np.zeros((nIn, nOut));
		self.costb   =  np.zeros(nOut);
		if activ == None:
			activ = "relu";
		self.actvat  = activ;
		self.indCache = [nIn, nOut];
	def calculate(self, inp):
		return np.array(inp).dot(self.weights)+self.biases;
	def relu(self, x):
		return np.maximum(x, 0);
	def sigmoid(self, x):
		return 1/(1+np.exp(-x));	
	def silu(self, x):
		return x/(1+np.exp(-x));
	def apply(self, inp):
		func = None;
		activ = self.actvat;
		if activ == None or activ == "relu":
			func = self.relu;
		elif activ == "sigmoid":
			func = self.sigmoid;
		elif activ == "silu":
			func = self.silu;
		return func(self.calculate(inp));
	def applygradients(self, lr):
		self.weights -= ( self.costw * lr );
		self.biases  -= ( self.costb * lr );

class Network:
	def __init__(self, size, activation=None):
		self.net = [];
		for i in range(len(size)-1):
			act = None;
			if activation != None and i < len(activation):
				act = activation[i];
			self.net.append(Layer(size[i], size[i+1], act));
	def calculate(self, inp):
		for i in range(len(self.net)):
			layer = self.net[i];
			inp = layer.apply(inp);
		return inp;
	def visnetwork(self):
		ind = 0;
		for i in self.net:
			print(f"\t{ind},{ind+1}");
			pp.pprint(i.weights);
			pp.pprint(i.biases);
			ind+=1;
	def mse(self, Y_hat, Y):
		n = Y_hat-Y;
		return n*n;
	def calccost(self, inp, out):
		cost = 0;
		for i in range(len(inp)):
			cost += self.mse(self.calculate(inp[i]), out[i]);
		return np.sum(cost/len(inp));
	def learn(self, x, y, lr, h = 0.00001):
		netcost = self.calccost(x,y);
		print("cost :",netcost);
		for layer in self.net:
			for n in range(layer.indCache[1]):
				layer.biases[n]+=h;
				dtcost = self.calccost(inps, outs) - netcost;
				layer.biases[n]-=h;
				layer.costb[n] = dtcost/h;

				for d in range(layer.indCache[0]):
					layer.weights[d][n] += h;
					dtcost = self.calccost(inps, outs) - netcost;
					layer.weights[d][n] -= h;
					layer.costw[d][n] = dtcost/h;
		for layer in self.net:
			layer.applygradients(lr);


nn = Network([2,12,12,2], ["relu","relu","sigmoid"]);
inps = [];
outs = [];
checkpoints = [
	[0   , .399],
	[.072, .51 ],
	[.157, .541],
	[.258, .533],
	[.34 , .487],
	[.399, .444],
	[.436, .393],
	[.466, .348],
	[.5  , .27 ],
	[.516, .196],
	[.539, .123],
	[.562, .076],
	[.565, 0   ]
];

i=0;
while i < 30:
	ran1 = np.round(random.uniform(0,1),1);
	ran2 = np.round(random.uniform(0,1),1);
	outside = False;
	
	for j in checkpoints:
		if ran1>=j[0] and ran2>=j[1]:
			outside = True;
			break;
	if outside:
		if outs.count([1,0]) > outs.count([0,1]):
			continue;
		outs.append([1,0]);
	else:
		if outs.count([0,1]) > outs.count([1,0]):
			continue;
		outs.append([0,1]);
	inps.append([ran1,ran2]);
	i+=1;
for i in range(len(inps)):
	c = 'red';
	if outs[i][0] == 1:
		c = 'blue';
	plt.scatter(inps[i][0], inps[i][1], color=c);
plt.show();

pp.pprint(inps);
pp.pprint(outs);
for i in range(len(inps)):
	c = 'red';
	if outs[i][0] == 1:
		c = 'blue';
	plt.scatter(inps[i][0], inps[i][1], color=c);
plt.show();

while True:
	try:
		nn.learn(inps, outs, .133);
		nn.visnetwork();
	except KeyboardInterrupt:
		break;
mylayout = np.zeros((500,500,3));
for y in range(500):
	print("Caculating graph... {:03d}%".format(int((y/499)*100)),end="\r");
	for x in range(500):
		nx = x/500;
		ny = 1-(y/500);
		pred = nn.calculate([nx,ny]);
		clr = [0,0,0];
		if pred[0]>pred[1]:
			clr = [0.8, .5, .5];
		else:
			clr = [.5, .5, .8];
		mylayout[y][x] = clr;

fig, ax = plt.subplots();
im = ax.imshow(mylayout);
for i in range(len(inps)):
	c = 'blue';
	if outs[i][0] == 1:
		c = 'red';
	ax.scatter(inps[i][0]*499, (1-inps[i][1])*499, color=c);
plt.show();
