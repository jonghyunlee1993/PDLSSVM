clc;
clear all;

addpath '/Users/jonghyun/Workspace/PDLSSVM/PDLSSVM_origin';

X = readtable('X.csv');
X = X{:, :};
y = readtable('y.csv');
y = y{:, :};

rho = 1 / 100;
c   = 100;
c1  = 0;
c2  = 0;

X_train = X([1:40, 51:90], :);
X_test  = X([41:50, 91:100], :);
y_train = y([1:40, 51:90], :);
y_test  = y([41:50, 91:100], :);

MAX_ITER = 500;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[m, n] = size(X_train);
e=ones(m,1);
alpha=rand(m,1);
beta=rand(m,1);
z=rand(n,1);
w=zeros(n,1);
u1=zeros(n,1);
u2=zeros(m,1);
u3=zeros(n,1);
t=0;
eps1=0.001;

Y    = diag(y_train);
H    = X_train * X_train';
B    = X_train' * Y;
I    = eye(m);
Iw   = eye(n);
Hw   = B * B';
ew   = ones(n,1);
temp = (1+2*rho)*Iw+c*Hw;
temp = inv(temp);

temp1 = inv(H+rho*(B'*B)+(1/c+rho)*I);

theta=1/2*(z-u1+B*beta-u3);

w = max( 0, c1/(2*rho)*ew - theta ) - max( 0, -c1/(2*rho)*ew - theta );
z=temp*(c*B*e+rho*w+rho*u1+rho*B*beta-rho*u3);

alpha = max( 0, c2/rho*e - beta-u2 ) - max( 0, -c2/rho*e - beta-u2 );
beta=temp1*rho*(alpha+u2+B'*z+B'*u3+1/rho*e);