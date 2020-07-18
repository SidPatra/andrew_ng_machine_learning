function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               
%U is 11x11, Z is 15x5, X_rec is 15x11
% There needs to be something in between that is 5x11
% X_rec = Z * [5x11] * U;
disp(size(U));
disp(size(Z));
disp(size(X_rec));

Ureduce = U(:,1:K); %same as before
disp(size(Ureduce)); %11x5

X_rec = Z * Ureduce'; % 15x5 x 5x11

% =============================================================

end