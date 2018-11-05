raw_tweets=tweets.message;
% raw_tweets=Mega1.cleantext;
cleanTextData = lower(raw_tweets);
cleanDocuments = tokenizedDocument(cleanTextData);
cleanDocuments = erasePunctuation(cleanDocuments);
cleanDocuments = removeWords(cleanDocuments, stopWords);
cleanDocuments = removeWords(cleanDocuments, ["newline"]);
cleanDocuments = removeWords(cleanDocuments, ["newlinenewline"]);
cleanDocuments = removeWords(cleanDocuments, ["utmsource"]);
cleanDocuments = removeWords(cleanDocuments, ["igtwittershare"]);
cleanDocuments = removeWords(cleanDocuments, ["httpswww"]);
cleanDocuments = removeWords(cleanDocuments, ["html"]);
cleanDocuments = removeWords(cleanDocuments, ["igshid"]);
cleanDocuments = removeShortWords(cleanDocuments,2);
cleanDocuments = removeLongWords(cleanDocuments,15);
cleanBag = bagOfWords(cleanDocuments);
cleanBag = removeInfrequentWords(cleanBag,2);

rawBag=bagOfWords(tokenizedDocument(raw_tweets));
figure
subplot(1,2,1)
wordcloud(rawBag);
title("Raw Data")
subplot(1,2,2)
wordcloud(cleanBag);
title("Clean Data")

%sentiment analysis
emb = fastTextWordEmbedding;
data = readLexicon;
numWords = size(data,1);
cvp = cvpartition(numWords,'HoldOut',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);
wordsTrain = dataTrain.Word;
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;
mdl = fitcsvm(XTrain,YTrain);
words = string(cleanBag.Vocabulary);
vec = word2vec(emb,words);
[YPred ,scores] = predict(mdl,vec);
scores(isnan(scores))=0;


figure
subplot(1,2,1)
idx = YPred == "Positive";
wordcloud(words(idx),scores(idx,1));
title("Predicted Positive Sentiment")

subplot(1,2,2)
wordcloud(words(~idx),scores(~idx,2));
title("Predicted Negative Sentiment")

%plot the word count
figure
wordcloud(cleanBag,'HighlightColor','red');
title('Words contained in the tweets')




% %find rural ect
% area_class = {'rural','poor','community','communitites','country','countryside','neighborhood',
%     'impoverished','poverty','broke','underpriviledged','low','income','low-income'};
% for i = 1:length(cleanDocuments)
%     tweet=string(joinWords(cleanDocuments));
%     dJobs(i) = contains(tweet{i},area_class,'IgnoreCase',true); 
% end
% idx=find(dJobs==1);
% idx_non=find(dJobs==0);
% for i=1:length(idx)
%     idxx=idx(i);
%     areaTweets{i}=tweet{idxx};
% end
% areaTweets = tokenizedDocument(areaTweets);
% areaTweets=bagOfWords(areaTweets);
% words = areaTweets.Vocabulary;
% vec = word2vec(emb,words);
% [~,scores_area] = predict(mdl,vec);
% scores_area(isnan(scores_area))=0;
% figure
% wordcloud(areaTweets,'HighlightColor','red');
% idx=find(dJobs==0);
% idx_non=find(dJobs==0);
% for i=1:length(idx)
%     idxx=idx(i);
%     nonareaTweets{i}=tweet{idxx};
% end
% nonareaTweets = tokenizedDocument(nonareaTweets);
% nonareaTweets=bagOfWords(nonareaTweets);
% words = nonareaTweets.Vocabulary;
% vec = word2vec(emb,words);
% [~,scores_nonarea] = predict(mdl,vec);
% scores_nonarea(isnan(scores_nonarea))=0;
% figure
% wordcloud(nonareaTweets,'HighlightColor','red');
% 
% 
