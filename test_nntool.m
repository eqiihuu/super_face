% This is a demo for Matlab nntools
clear all;
clc;

P=[110 0.807 240 0.2 15 1 18 2 1.5;
110 2.865 240 0.1 15 2 12 1 2;
110 2.59 240 0.1 12 4 24 1 1.5;
220 0.6 240 0.3 12 3 18 2 1;
220 3 240 0.3 25 3 21 1 1.5;
110 1.562 240 0.3 15 3 18 1 1.5;
110 0.547 240 0.3 15 1 9 2 1.5];

T=[54248 162787 168380 314797;
28614 63958 69637 82898;
86002 402710 644415 328084;
230802 445102 362823 335913;
60257 127892 76753 73541;
34615 93532 80762 110049;
56783 172907 164548 144040];
m=max(max(P));
n=max(max(T));
P=P'/m;
T=T'/n;
%-------------------------------------------------------------------------%
pr(1:9,1)=0; % Range matrix of input vector
pr(1:9,2)=1;
bpnet=newff(pr,[12 4],{'logsig', 'logsig'}, 'traingdx', 'learngdm');
% Build the BP neural network? 12 hidden neuron?4 output neuron
%tranferFcn attribute 'logsig', use Sigmoid for hidden layer
%tranferFcn attribute 'logsig', use Sigmoid for output layer
%trainFcn attribute 'traingdx', auto-adjest learning rate, momentum, gradient decent, backpropagation
%learn attribute 'learngdm', momentum, gradient decent
net.trainParam.epochs=1000;% maximium training steps: 2000
net.trainParam.goal=0.001; % minimum error 0.001
net.trainParam.show=10; % show result every 100 steps
net.trainParam.lr=0.05; % learning rate: 0.05
bpnet=train(bpnet,P,T);
%-------------------------------------------------------------------------
p=[110 1.318 300 0.1 15 2 18 1 2];
p=p'/m;

r=sim(bpnet,p);
R=r'*n;
display(R);