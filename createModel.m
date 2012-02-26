
load('smap.mat', 'smap');
load('smapStemmedUnique.mat', 'smapUnique', 'uniqToSmap', 'smapToUniq', 'smapStemmed');
load('stopwords.mat', 'swords', 'swordIndexes');
swordIndexes = cell2mat(swordIndexes);

dictLen = length(smap);
numTokens = 20000000; % each item is 4 bytes. read multiple of 3 each time since we have 3 tokens
X = []; % NxM matrix. N is num terms, M is num docs.
y = []; % Nx1 matrix. Contains ratings


ratingIndex = strmatch('<rating>', smap, 'exact');
reviewTextIndex = strmatch('<review_text>', smap, 'exact');
endReviewTextIndex = strmatch('</review_text>', smap, 'exact');
endReviewIndex = strmatch('</review>', smap, 'exact');

seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');

f = fopen('tokens.bin');
partition = 1;
Xtot = [];
ytot = [];
while ~feof(f)
%for i=1:1 
  buf = fread(f, numTokens*3, 'int32');
  buf = buf(3:3:end);
  reviewTextEnds = find(buf == endReviewTextIndex);
  reviewEnds = find(buf == endReviewIndex);
  buf = buf(1:reviewEnds(end)); % stop at last whole review
  fseek(f, -(numTokens - reviewEnds(end))*4*3,'cof'); % seek to end of last whole review
  reviewTextStarts = find(buf == reviewTextIndex);
  ratingStarts = find(buf == ratingIndex);

  uniques = logical([]); % indexes of unique reviews

  y = cell2mat(smap(buf(ratingStarts + 1))) - '0'; % Get all ratings and convert them into numerical vector with values 1-5
  numReviews = length(reviewTextStarts);
  tic;
  X = sparse(1 + dictLen, numReviews);
  for i = 1:numReviews
    text = buf(reviewTextStarts(i) + 1:reviewTextEnds(i) - 1);
    key = mat2str(text(1:min(10,end)));
    if isKey(seen, key)
      continue
    else
      seen(key) = true;
      uniques = [uniques; i];
    end
    
    % omit stopwords
    text = setdiff(text, swordIndexes);

    % translate to stemmed indexes
    revLen = length(text);
    for k=1:revLen
        text(k) = smapToUniq(text(k));
    end
    
    review = sparse(text, 1, 1, dictLen, 1);
    X(:,i) = [1; review];
    if mod(i, 10000) == 0
      display('done 10000: ')
      toc
      tic;
    end
  end
  toc;
  if length(uniques) == 0 % Hit EOF
    break
  end
  display('Num uniques: ')
  display(length(uniques))
  Xtot = [Xtot, X(:,uniques)];
  ytot = [ytot; y(uniques)];
  partition = partition + 1
end
save('model','Xtot','ytot');
fclose(f)
