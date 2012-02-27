%modelFileName = 'model'; % should have vars Xtot and ytot
modelFileName = 'modelNoStopsStemmed';
load(modelFileName);
numWords = 10000;
numReviews = length(Xtot(1,:));

% Get total frequency for each unique stem
uniqWordTotals = sum(Xtot, 2);
[ignore, uniquesByFreq] = sort(uniqWordTotals, 'descend');

stopwordsUniq = unique(smapToUniq(swordIndexes), 'first');
stopwordsLen = length(stopwordsUniq);

% Omit stemmed stopwords
[ignore, ix] = setdiff(uniquesByFreq, stopwordsUniq);
uniquesNoStops = uniquesByFreq(sort(ix)); % keep freq order

% top stems by frequency
topWords = uniquesNoStops(1:numWords);

iterations = 1
lambda = 1; % regularization constant
rng(13);
ri = randperm(numReviews);
Xtot = Xtot(:,ri); % shuffle reviews
ytot = ytot(ri); % shuffle reviews
aucs = [];
lifts = [];
times = [];
flops = [];

for i = 1:iterations
  % separate data into training and validation
  validationStart = numReviews*(i-1)/10 + 1;
  validationEnd = numReviews*i/10;
  trainingX = [Xtot(topWords, 1:validationStart-1), Xtot(topWords, validationEnd:end)];
  validationX = Xtot(topWords, validationStart:validationEnd);
  trainingY = [ytot(1:validationStart-1); ytot(validationEnd:end)];
  validationY = ytot(validationStart:validationEnd);
  
  % get B from regression
  r = speye(numWords)*lambda;
  tic;
  B = (trainingX * trainingX' + r)\(trainingX*trainingY);
  times(i) = toc
  
  % calculate flops
  % nza = double(trainingX ~= 0);
  % nzb = double(trainingX' ~= 0);
  % ftmp = nza*nzb;
  % ftmp = 2*ftmp - (ftmp ~= 0);
  % f = sum(sum(ftmp));
  % flops(i) = f;

  scores = validationX'*B;
  [x,y,t,AUC] = perfcurve((validationY > 3),scores,true);
  aucs = [aucs, AUC];
  [l, index] = min(abs(x-.01));
  lifts = [lifts, y(index)/.01];
  display(i)
end
% perf = flops ./ times;
