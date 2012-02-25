numWords = 10000;
numReviews = length(X(1,:));
rng(13);
c = 1 % regularization constant
X = X(:,randperm(numReviews)); % shuffle reviews

for i = 1:10
  % separate data into training and validation
  validationStart = numReviews*(i-1)/10 + 1;
  validationEnd = numReviews*i/10;
  trainingX = [Xtot(1:numWords, 1:validationStart-1), Xtot(1:numWords, validationEnd:end)];
  validationX = Xtot(1:numWords, validationStart:validationEnd);
  trainingY = [ytot(1:validationStart-1); ytot(validationEnd:end)];
  validationY = ytot(validationStart:validationEnd);
  
  % get B from regression
  r = speye(numReviews)*c;
  B = (trainingX * trainingX' + r)\(trainingX*trainingY);
  scores = validationX'*B;
  [x,y,AUC] = perfcurve((validationY > 3), (scores > 3),1);
end