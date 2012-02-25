modelFileName = 'model'; % should have vars Xtot and ytot
load(modelFileName);
numWords = 10000;
numReviews = length(Xtot(1,:));

iterations = 1
c = 1; % regularization constant
rng(13);
ri = randperm(numReviews);
Xtot = Xtot(:,ri); % shuffle reviews
ytot = ytot(ri); % shuffle reviews
aucs = []
for i = 1:iterations
  % separate data into training and validation
  validationStart = numReviews*(i-1)/10 + 1;
  validationEnd = numReviews*i/10;
  trainingX = [Xtot(1:numWords, 1:validationStart-1), Xtot(1:numWords, validationEnd:end)];
  validationX = Xtot(1:numWords, validationStart:validationEnd);
  trainingY = [ytot(1:validationStart-1); ytot(validationEnd:end)];
  validationY = ytot(validationStart:validationEnd);
  
  % get B from regression
  r = speye(numWords)*c;
  B = (trainingX * trainingX' + r)\(trainingX*trainingY);
  scores = validationX'*B;
  [x,y,t,AUC] = perfcurve((validationY > 3),scores,true);
  aucs = [aucs, AUC];
  display(i)
end