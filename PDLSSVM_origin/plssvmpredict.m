function [predictY,sparseprimal] = plssvmpredict(testX,w,z,v)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
predictY=[];
[m,n]=size(testX);

if z==0
    predictY=sign(testX*w(1:end-1)+w(end));  %testX������������ֱ�߾���
else
    predictY=sign(testX*z+w(end));  %testX������������ֱ�߾���
end
sparseprimal=sum(z==0);
