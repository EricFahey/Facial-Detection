% you might want to have as many negative examples as positive examples
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
mkdir(new_imageDir);
n_have = numel(dir('cropped_training_images_notfaces/*.jpg'));

dim = 36;
img_idx = 209;

while n_have < n_want*2
    
    % generate random 36x36 crops from the non-face images
    img = im2single(rgb2gray(imread(sprintf('%s/%s', imageDir, imageList(img_idx).name)))); 
    [img_y, img_x] = size(img);
    
    cropped_img = img(randi(img_y - dim + 1) + (0 : dim - 1), ...
        randi(img_x - dim + 1) + (0 : dim - 1));
    n_have = n_have + 1;
    imwrite(cropped_img, sprintf('%s/%d.jpg', new_imageDir, n_have), 'jpg');
    
    %est_x = floor(img_x/dim);
    %est_y = floor(img_y/dim);
    %for i = 1 : est_y
    %    for k = 1 : est_x
    %        
    %        cropped_img = img((i - 1) * dim + 1 : (i - 1) * dim + dim, ...
    %            (k - 1) * dim + 1 : (k - 1) * dim + dim);
    %        n_have = n_have + 1;
    %        imwrite(cropped_img, sprintf('%s/%d.jpg', new_imageDir, n_have), 'jpg');
    %    end
    %end
    
    
    %increment idx by 1 or reset back to 1
    if (img_idx >= nImages)
        img_idx = 1;
    else
        img_idx = img_idx + 1;
        %fprintf('Image Index: %d/%d\n', img_idx, nImages);
    end
end

fprintf('Successfully generated %d negatives!\n', n_have);