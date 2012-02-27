load('smap.mat','smap');
dictLen = length(smap);

% Stem the words in the smap dictionary
%smapStemmed = cellfun(@(x) porterStemmer(x), smap, 'UniformOutput', false);
%for i=1:dictLen
%    smapStemmedShort{i} = porterStemmer(smap{i});
%    if mod(i, 10000) == 0
%      display('done 10000: ')
%      i
%    end
%end
%save('smapStemmed.mat', 'smapStemmed');

% Stemming the words will cause duplicates in the dictionary, so get uniques
% Generate: unique words dictionary, map of unique to smap indexes, map of smap to unique indexes
load('smapStemmed.mat', 'smapStemmed');
[smapUnique, uniqToSmap, smapToUniq] = unique(smapStemmed, 'first');
save('smapStemmedUnique.mat', 'smapUnique', 'uniqToSmap', 'smapToUniq', 'smapStemmed');

