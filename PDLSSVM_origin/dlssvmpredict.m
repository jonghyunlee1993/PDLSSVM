function [predictY,sparsedual] = dlssvmpredict(X,Y,testX,alpha,z,v)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predictY=[];
[m,n]=size(testX);
Y=diag(Y);
if z==0
    w=alpha'*Y*X;
    predictY=sign(testX*w');  %testX各样本到正类直线距离
else
    w=z'*Y*X;
    predictY=sign(testX*w');  %testX各样本到正类直线距离
end
sparsedual=sum(z<=0.001);
