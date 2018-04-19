run('../vlfeat-0.9.20/toolbox/vl_setup')
load('pos_neg_feats.mat')

%split negative features into 80% training set and 20% validation set
neg_last_row = int32(floor(0.8 * neg_nImages));
neg_training_feats = neg_feats(1 : neg_last_row, :);
neg_validation_feats = neg_feats(neg_last_row + 1 : end, :);

%split positive features into 80% training set and 20% validation set
pos_last_row = int32(floor(0.8 * pos_nImages));
pos_training_feats = pos_feats(1 : pos_last_row, :);
pos_validation_feats = pos_feats(pos_last_row + 1 : end, :);

training_feats = cat(1, pos_training_feats, neg_training_feats);
training_labels = cat(1, ones(pos_last_row, 1), -1 * ones(neg_last_row, 1));

lambda = 0.0001;
[weight,bias] = vl_svmtrain(training_feats',training_labels',lambda);

save('my_svm.mat','weight','bias');

fprintf('Classifier performance on train data:\n')
confidences = [pos_training_feats; neg_training_feats]*weight + bias;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, training_labels);

%5. Test SVM with validation feature set
validation_labels = cat(1, ones(pos_nImages - pos_last_row, 1), -1 * ones(neg_nImages - neg_last_row, 1));
fprintf('Classifier performance on validation data:\n')
confidences = [pos_validation_feats; neg_validation_feats]*weight + bias;
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, validation_labels);

