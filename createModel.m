
load('smap.mat', 'smap');
dictLen = length(smap);
readLength = 60000000; % each item is 4 bytes. read multiple of 3 each time since we have 3 tokens

X = []; % NxM matrix. N is num terms, M is num docs.
y = []; % Nx1 matrix. Contains ratings


ratingIndex = strmatch('<rating>', smap, 'exact');
reviewTextIndex = strmatch('<review_text>', smap, 'exact');
endReviewTextIndex = strmatch('</review_text>', smap, 'exact');
nextTokenIsRating = false;
insideReviewText = false;

out = [];

f = fopen('tokens.bin');
partition = 1;
c = 0;
times = []
%while ~feof(f)
for j = 1:1
  buf = fread(f, readLength, 'int32');
  buf = buf(3:3:end);
  reviewStarts = find(buf == reviewTextIndex);
  reviewEnds = find(buf == endReviewTextIndex);
  
  if length(reviewStarts) > length(reviewEnds) 
    reviewStarts = reviewStarts(1:end-1);
  end
  ratingStarts = find(buf == ratingIndex);
  y = cell2mat(smap(buf(ratingStarts + 1))) - '0'; % Get all ratings and convert them into numerical vector (1-5)
  numReviews = length(reviewStarts)
  tic;
  X = sparse(1 + dictLen, numReviews)
  for i = 1:numReviews
    text = buf(reviewStarts(i) + 1:reviewEnds(i) - 1);
    review = sparse(text, 1, 1, dictLen, 1);
    X(:,i) = [1; review];
    if mod(i, 1000) == 0
      display('done 1000: ')
      toc
      tic;
    end
  end
  save(strcat('res',int2str(partition)), 'X', 'y')
  partition = partition + 1
end

fclose(f)

% B = (X * X' + eye(length(X)))\(X*y)
% score = X'*B
% find((score > 3) == (y > 3))