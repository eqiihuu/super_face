%% Load Image Information from face Directory
faceDatabase = imageSet('../data/2D_gender','recursive');

%% Display Montage of First Gender
figure;
montage(faceDatabase(1).ImageLocation);
title('Images of Single Gender');

%%  Display Query Image and Database Side-Side
genderToQuery = 1; % 1 is female, 2 is male
galleryImage = read(faceDatabase(genderToQuery),1); % first gender image
for i=1:size(faceDatabase,2)
    imageList(i) = faceDatabase(i).ImageLocation(5); % gender image to match
end
figure; 
title('One of the images from each gender subfolder')
subplot(1,2,1);imshow(galleryImage);
subplot(1,2,2);montage(imageList);
diff = zeros(1,9);

%% Split Database into Training & Test Sets
[training,test] = partition(faceDatabase,[0.8 0.2]);


%% Extract and display Histogram of Oriented Gradient Features for single face 
person = 1;
[hogFeature, visualization]= ...
    extractHOGFeatures(int16(read(training(person),1)));
figure;
subplot(2,1,1);imshow(read(training(person),1));title('Input Face');
subplot(2,1,2);plot(visualization);title('HoG Feature');

%% Extract HOG Features for training set 
featureVectorLen = size(hogFeature,2) % 142884
trainingFeatures = zeros(size(training,2)*training(1).Count,featureVectorLen);
featureCount = 1;
for i=1:size(training,2)
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(int16(read(training(i),j)));
        trainingLabel{featureCount} = training(i).Description;    
        featureCount = featureCount + 1;
    end
    personIndex{i} = training(i).Description;
end

%% Create 40 class classifier using fitcecoc 
faceClassifier = fitcecoc(trainingFeatures,trainingLabel);


%% Test Images from Test Set 
person = 1;
queryImage = read(test(person),1);
queryFeatures = extractHOGFeatures(int16(queryImage));
personLabel = predict(faceClassifier,queryFeatures);
% Map back to training set to find identity 
booleanIndex = strcmp(personLabel, personIndex);
integerIndex = find(booleanIndex);
figure;
subplot(1,2,1);imshow(queryImage);title('Query Face');
subplot(1,2,2);imshow(read(training(integerIndex),1));title('Matched Class');

%% Test
% figure;
count = 0
for person=2:2 % 1 is female, 2 is male
    for j = 1:test(person).Count
        queryImage = read(test(person),j);
        queryFeatures = extractHOGFeatures(int16(queryImage));  % Extract HOG feature
        personLabel = predict(faceClassifier,queryFeatures); % classification (?)
        disp(personLabel)
        % Map back to training set to find identity
        booleanIndex = strcmp(personLabel, personIndex);
        disp(booleanIndex)
        integerIndex = find(booleanIndex);
%         subplot(2,2,figureNum);imshow(imresize(queryImage,3));title('Query Face');
%         subplot(2,2,figureNum+1);imshow(imresize(read(training(integerIndex),1),3));title('Matched Class');
%         figureNum = figureNum+2;
        if (person == 1 && strcmp(personLabel,'female')) || (person == 2 && strcmp(personLabel,'male'))
            count = count + 1;
        end
        
    end
    disp('RESULT Percent:')
    result = count / test(person).Count * 100.00
    disp(result)
%     figure;
%     figureNum = 1;

end



