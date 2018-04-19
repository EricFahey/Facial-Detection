close all
clear
run('../vlfeat-0.9.20/toolbox/vl_setup')

pos_imageDir = 'cropped_training_images_faces';
pos_imageList = dir(sprintf('%s/*.jpg',pos_imageDir));
pos_nImages = length(pos_imageList);

neg_imageDir = 'cropped_training_images_notfaces';
neg_imageList = dir(sprintf('%s/*.jpg',neg_imageDir));
neg_nImages = length(neg_imageList);

cellSize = 6;
dim = 36;
featSize = 31*(dim/cellSize)^2;
img_idx = 1;

pos_feats = zeros(pos_nImages*2,featSize);
for i=1:pos_nImages
    img = im2single(imread(sprintf('%s/%s',pos_imageDir,pos_imageList(img_idx).name)));
    flipped_img = fliplr(img);
    
    feat = vl_hog(img, cellSize);
    pos_feats(i,:) = feat(:);
    
    %also generate features for flipped image
    flipped_feat = vl_hog(flipped_img, cellSize);
    pos_feats(i + pos_nImages,:) = flipped_feat(:);
    
    fprintf('got feat for pos image %d/%d\n',i,pos_nImages);
    %increment idx by 1 or reset back to 1
    if (img_idx >= pos_nImages)
        img_idx = 1;
    else
        img_idx = img_idx + 1;
    end
end
%technically now there's x2 images
pos_nImages = pos_nImages*2;

neg_feats = zeros(neg_nImages,featSize);
for i=1:neg_nImages
    im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name)));
    feat = vl_hog(im,cellSize);
    neg_feats(i,:) = feat(:);
    fprintf('got feat for neg image %d/%d\n',i,neg_nImages);
end

save('pos_neg_feats.mat','pos_feats','neg_feats','pos_nImages','neg_nImages')