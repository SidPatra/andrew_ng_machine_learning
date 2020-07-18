function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
idx = X(1)-centroids(1);

disp('centroid dimensions');
disp(size(centroids)); % 5 row, 11 col
%K = 5

for i=1:size(X)(1)
  temp = ones(K,1); %creates k dimensional vector - will store squared error 
                    %losses for each centroid
  for j=1:K
    temp(j)=(1/size(X)(2)) * sqrt(sum((X(i,:)-centroids(j,:)).^2)); %it's actualy square root of sum
    %temp(j) = (1/size(X)(2)) * sum(sqrt((X(i,:)-centroids(j,:)).^2)); %sum of squared differences
  endfor
  disp('temp');
  disp(temp);
  idx(i) = find(temp==min(temp(:))); % index of centroid closest to example i
endfor




% =============================================================

end

