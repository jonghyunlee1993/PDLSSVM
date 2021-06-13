clc;
clear all;

addpath '/Users/jonghyun/Workspace/PDLSSVM/PDLSSVM_origin';

X = readtable('data/X_gisette.csv');
X = X{:, :};
y = readtable('data/y_gisette.csv');
y = y{:, :};

X_train = X([1:5900], :);
X_test  = X([5901:6000], :);
y_train = y([1:5900], :);
y_test  = y([5901:6000], :);


% SPDLSSVM
rho = 1 / 100;
c   = 100;
c1  = 1 / 1000;
c2  = 1 / 1000;

[w,alpha,z,beta] = pdlssvm(X_train, y_train, rho, c, c1, c2);
[predictY, sparseprimal, sparsedual] = pdlssvmpredict(X_train, y_train, X_test, w, alpha, z, beta, 1);
sum(predictY == y_test)


% LSSVM
rho = 1 / 100;
c   = 100;
c1  = 0;
c2  = 0;

[w,alpha,z,beta] = pdlssvm(X_train, y_train, rho, c, c1, c2);
[predictY, sparseprimal, sparsedual] = pdlssvmpredict(X_train, y_train, X_test, w, alpha, z, beta, 1);
sum(predictY == y_test)

