run('../vlfeat-0.9.20/toolbox/vl_setup');

%load SVM trained in previous part
load('my_svm.mat');

cell_size = 6;
dimension = 36;
feature_size = 31 * (dimension / cell_size)^2;

bound_boxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

scales = [1, 0.75, 0.50, 0.25, 0.15];
scales_size = size(scales, 2);

class_image_color = imread('class.jpg');

class_image = im2single(rgb2gray(class_image_color));

%generate features for various scaled images
for i = 0 : 32
    scale = 1 - (0.025 * i);
    %resize image
    scaled_image = imresize(class_image, scale);
    
    % generate a grid of features across the entire image. you may want to 
    % try generating features more densely (i.e., not in a grid)
    features = vl_hog(scaled_image, cell_size);

    % concatenate the features into 6x6 bins, and classify them (as if they
    % represent 36x36-pixel faces)
    [rows,cols,~] = size(features);    
    confs = zeros(rows - cell_size + 1, cols - cell_size + 1);
    for r=1:rows - cell_size + 1
        for c=1:cols - cell_size + 1

            feature_vector = features(r:r + cell_size - 1, c:c + cell_size - 1, :);

            % create feature vector for the current window and classify it using the SVM model, 
            % take dot product between feature vector and w and add b,
            % store the result in the matrix of confidence scores confs(r,c)
            confs(r,c) = feature_vector(:)'*weight + bias;
        end
    end
    % get the most confident predictions 
    [~,inds] = sort(confs(:),'descend');
    recall_num = 40;
    if (size(inds, 1) < recall_num)
        recall_num = size(inds, 1);
    end
    inds = inds(1:recall_num); % (use a bigger number for better recall)
    for n=1:numel(inds)        
        [row, col] = ind2sub([size(confs,1) size(confs,2)],inds(n));
        confidence = confs(row, col);
        %filter out under 0.90 confidence
        if confidence < 0.90
            continue
        end
        
        scale_factor = 1 / scale;

        bound_box = [ col*cell_size*scale_factor ...
                 row*cell_size*scale_factor ...
                (col+cell_size-1)*cell_size*scale_factor ...
                (row+cell_size-1)*cell_size*scale_factor];
        image_name = 'class.jpg';
        % save         
        bound_boxes = [bound_boxes; bound_box];
        confidences = [confidences; confidence];  
        image_names = [image_names; image_name];
           
    end
end

%low-maximum supression
detection_count = size(confidences, 1);
%resort confidences since higher confidences should be boxed first
[confidences, index] = sort(confidences, 'descend');
bound_boxes = bound_boxes(index, :);
%boolean array to specify which bound boxes to draw
valid_bound_boxes = zeros(detection_count, 1);

for i = 1 : detection_count
    bound_box = bound_boxes(i, :);
    %assume each confidence is valid unless otherwise specified
    is_valid = true;
    
    %check previously defined boxes
    for j = 1 : detection_count
        if (valid_bound_boxes(j) == true)
            other_bound_box = bound_boxes(j, :);
            left = max(bound_box(1), other_bound_box(1));
            top = max(bound_box(2), other_bound_box(2));
            right = min(bound_box(3), other_bound_box(3));
            bottom = min(bound_box(4), other_bound_box(4));
            if (left <= right && top <= bottom)
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
    valid_bound_boxes(i) = is_valid;
end

%lets plot our bound boxes
figure(1);
imshow(class_image_color);
hold on

for i = 1 : detection_count
    if (valid_bound_boxes(i) == 1)
        %fprintf('Valid Index: %d\n', i);
        bound_box = bound_boxes(i, :);
        plot_rectangle = [bound_box(1), bound_box(2); ...
            bound_box(1), bound_box(4); ...
            bound_box(3), bound_box(4); ...
            bound_box(3), bound_box(2); ...
            bound_box(1), bound_box(2)];
        plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
    end
end


