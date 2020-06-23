% driver function
function [optTehta] = costDriver()
options = optimset('GradObj','on','MaxIter','100'); % specifying options of types of optimization here in a special data structure.
initialTheta = zeros(2,1); %intial theta needs to be at least dimension 2.
[optTheta, functionVal, exitFlag] = fminunc(@costFunction,initialTheta,options)
% @ denotes pointer to function. 
% fminunc is used to solve unconstrained optimization problems.
% there are many different options that we can use.
% read up on fminunc later.