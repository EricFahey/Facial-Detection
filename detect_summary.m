% CPS843/CP8307 Assignment 3 - Filtering and Machine Learning
% Student Name: Eric Fahey
% Student Number: 500641389
% Student Name: Victor Huynh
% Student Number: 500634673

fprintf('In our detector, we again took 6x6 feature blocks of our testing image\n');
fprintf('and then classified it using the SVM we trained in part 1.\n');
fprintf('We would then take the highest confidences over a threshold of 0.9\n');
fprintf('and perform non-maximum suppression to elimate most overlapping boxes.\n');
fprintf('We chose an overlapping threshold of 0.2 for our non-maximum suppresion.\n');
fprintf('We would repeat this process for multiple scaled versions of the testing image.\n');
fprintf('We also implemented a script called detect_class_faces.m which would\n');
fprintf('plot bounding boxes around faces found in the image.\n');
fprintf('We were able to achieve an accuracy of 0.329, however this is likely due to\n');
fprintf('false positives. We tried to improve performance by implementing\n');
fprintf('Hard Negative Mining, however it ended up making performance worse.\n');
fprintf('In addition, we added mirrored faces to our dataset to improve performance.\n');
fprintf('Our performance on class.jpg is good, however we were unable to pick up some faces,\n');
fprintf('primarily with darker skinned individuals. It may be necessary to add additional\n');
fprintf('positives to our dataset. We also had an issue with false positives.')