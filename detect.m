run('../vlfeat-0.9.20/toolbox/vl_setup')

imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

%load SVM trained in previous part
load('my_svm.mat');

cell_size = 6;
dimension = 36;
feature_size = 31 * (dimension / cell_size)^2;

bound_boxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

scales = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, ...
    0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05];
scales_size = size(scales, 2);

for i = 1 : nImages
    close all
    colored_image = imread(sprintf('%s/%s', imageDir, imageList(i).name));

    %it seems some images are already grayscale
    image = colored_image;
    if (size(colored_image, 3) > 1)
        image = rgb2gray(colored_image);
    end
    image = im2single(image);
    
    img_bound_boxes = zeros(0,4);
    img_confidences = zeros(0,1);
    img_image_names = cell(0,1);
    
    %generate features for various scaled images
    for j = 1 : scales_size

        scale = scales(j);
        %resize image
        scaled_image = imresize(image, scale);

        % generate a grid of features across the entire image. you may want to 
        % try generating features more densely (i.e., not in a grid)
        features = vl_hog(scaled_image, cell_size);

        % concatenate the features into 6x6 bins, and classify them (as if they
        % represent 36x36-pixel faces)
        [rows,cols,~] = size(features);    
        current_confidences = zeros(rows - cell_size + 1, cols - cell_size + 1);
        for r=1:rows - cell_size + 1
            for c=1:cols - cell_size + 1

                feature_vector = features(r:r + cell_size - 1, c:c + cell_size - 1, :);

                % create feature vector for the current window and classify it using the SVM model, 
                % take dot product between feature vector and w and add b,
                % store the result in the matrix of confidence scores confs(r,c)
                current_confidences(r,c) = feature_vector(:)'*weight + bias;
            end
        end
        % get the most confident predictions 
        [~,inds] = sort(current_confidences(:),'descend');
        recall_num = 40;
        if (size(inds, 1) < recall_num)
            recall_num = size(inds, 1);
        end
        inds = inds(1:recall_num); % (use a bigger number for better recall)
        for n=1:numel(inds)        
            [row, col] = ind2sub([size(current_confidences,1) size(current_confidences,2)],inds(n));
            confidence = current_confidences(row, col);
            %filter out under 0.90 confidence
            if confidence < 0.90
                continue
            end

            scale_factor = 1 / scale;

            bound_box = [ col*cell_size*scale_factor ...
                     row*cell_size*scale_factor ...
                    (col+cell_size-1)*cell_size*scale_factor ...
                    (row+cell_size-1)*cell_size*scale_factor];
            image_name = {imageList(i).name};
            % save         
            img_bound_boxes = [img_bound_boxes; bound_box];
            img_confidences = [img_confidences; confidence];  
            img_image_names = [img_image_names; image_name];
        end
    end

    %low-maximum supression
    detection_count = size(img_confidences, 1);
    %resort confidences since higher confidences should be boxed first
    [img_confidences, index] = sort(img_confidences, 'descend');
    img_bound_boxes = img_bound_boxes(index, :);
    %boolean array to specify which bound boxes to draw
    valid_bound_boxes = zeros(detection_count, 1);

    for j = 1 : detection_count
        bound_box = img_bound_boxes(j, :);
        %assume each confidence is valid unless otherwise specified
        is_valid = true;

        %check previously defined boxes
        for k = 1 : detection_count
            if (valid_bound_boxes(k) == 1)
                other_bound_box = img_bound_boxes(k, :);
                left = max(bound_box(1), other_bound_box(1));
                top = max(bound_box(2), other_bound_box(2));
                right = min(bound_box(3), other_bound_box(3));
                bottom = min(bound_box(4), other_bound_box(4));
                if (left < right && top < bottom)
                    intersection_width = right - left + 1;
                    %fprintf('Width: %d\n', intersection_width);
                    intersection_height = bottom - top + 1;
                    %fprintf('Height: %d\n', intersection_height);
                    intersection_area = intersection_width * intersection_height;
                    bound_box_area = (bound_box(3) - bound_box(1) + 1) ...
                        * (bound_box(4) - bound_box(2) + 1);
                    other_bound_box_area = (other_bound_box(3) - other_bound_box(1) + 1) ...
                        * (other_bound_box(4) - other_bound_box(2) + 1);
                    union_area = bound_box_area + other_bound_box_area - intersection_area;
                    overlap = intersection_area / union_area;
                    %fprintf('Overlap: %d\n', overlap);
                    if (overlap > 0.20)
                        is_valid = false;
                    end

                    if(other_bound_box(1) <= (bound_box(1) + bound_box(3))/2 ...
                            && (bound_box(1) + bound_box(3))/2 <= other_bound_box(3) ...
                            && other_bound_box(2) <= (bound_box(2) + bound_box(4))/2 ...
                            && (bound_box(2) + bound_box(4))/2 <= other_bound_box(4))
                        %fprintf('Center!\n');
                        is_valid = false;
                    end
                end

            end
        end
        valid_bound_boxes(j) = is_valid;
    end
    
    %show current image
    %figure(1)
    %imshow(colored_image);
    %hold on
    
    %lets plot our bound boxes
    for j = 1 : detection_count
        if (valid_bound_boxes(j) == 1)
            %fprintf('Valid Index: %d\n', j);
            bound_box = img_bound_boxes(j, :);
            plot_rectangle = [bound_box(1), bound_box(2); ...
                bound_box(1), bound_box(4); ...
                bound_box(3), bound_box(4); ...
                bound_box(3), bound_box(2); ...
                bound_box(1), bound_box(2)];
            %plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
        end
    end
    % save         
    bound_boxes = [bound_boxes; img_bound_boxes];
    confidences = [confidences; img_confidences];  
    image_names = [image_names; img_image_names];
    fprintf('got predictions for image %d/%d (%s), press enter to continue...\n', i, nImages, imageList(i).name);
    %pause;
end

% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bound_boxes, confidences, image_names, label_path);


