%modelFileName = 'model'; % should have vars Xtot and ytot
%modelFileName = 'modelNoStops';
%modelFileName = 'modelNoStopsStemmed';
modelFileName = 'modelStemmed';

load(modelFileName);
%load('stopwords');
load('smapStemmedUnique');
load('smap', 'smap');
display('loaded data')

numWords = 10000;
numReviews = length(Xtot(1,:));

% Get total frequency for each unique stem
uniqWordTotals = sum(Xtot(2:end,:), 2);
uniqWordTotals = uniqWordTotals(1:length(uniqToSmap), 1);
[ignore, uniquesByFreq] = sort(uniqWordTotals, 'descend');

wordTotals = sum(Xtot(2:end,:), 2);
[ignore, wordsByFreq] = sort(wordTotals, 'descend');
wordsByFreq = [1; wordsByFreq+1];

% top stems by frequency
%topWords = uniquesNoStops(1:numWords) + 1;
topWords = wordsByFreq(1:numWords);
%topWords = 1:numWords;

iterations = 1;
%lambda = 1; % regularization constant
lambda = 1500; % regularization constant
rng(13);
ri = randperm(numReviews);
Xtot = Xtot(:,ri); % shuffle reviews
ytot = ytot(ri); % shuffle reviews
aucs = [];
lifts = [];
times = [];
flops = [];
Bs = [];

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
  Bs = [Bs B];
  display(i)
end
meanB = mean(Bs,2);


% perf = flops ./ times;

% stemmed
%[a, ix] = sort(B(2:end));
%[a, ix] = sort(meanB(2:end));
%smap(uniqToSmap(topWords(ix(1:20)+1)))
%smap(uniqToSmap(topWords(ix(numWords-20:end)+1)))

%smapUnique(topWords(ix(1:20)+1))'
%smapUnique(topWords(ix(numWords-20:end)+1))'
%a(1:20) % lowest weights
%a(numWords-20:end) % highest weights

%unstemmed
% worst words:
%[a, ix] = sort(B(2:end));
%smap(ix(1:20))
%smap(ix(numWords-20:end))

%[a,ix] = sort(B(2:end), 'descend');
%smap(wordsByFreq(ix(1:20)))

