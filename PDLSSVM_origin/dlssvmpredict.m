function [predictY,sparsedual] = dlssvmpredict(X,Y,testX,alpha,z,v)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predictY=[];
[m,n]=size(testX);
Y=diag(Y);
if z==0
    w=alpha'*Y*X;
    predictY=sign(testX*w');  %testX������������ֱ�߾���
else
    w=z'*Y*X;
    predictY=sign(testX*w');  %testX������������ֱ�߾���
end
sparsedual=sum(z<=0.001);
