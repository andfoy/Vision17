setup;

load model.mat

load data/wider_face_split/wider_face_train.mat

hogCellSize = 8 ;
minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
scales = 2.^linspace(...
  minScale,...
  maxScale,...
  numOctaveSubdivisions*(maxScale-minScale+1)) ;


for e=1:numel(event_list)
    event = event_list(e);
    imPath = fullfile('data', 'WIDER_val', 'images', event);
    files = fullfile(imPath{1}, '*.jpg');
    img_names = dir(files);
    files = fullfile(imPath{1}, {img_names.name});
    for f = 1:numel(files)
        fprintf('Processing: %s\n', img_names(f).name);
        img = imread(files{f});
        img = im2single(img);
        scores = [0];
        detections = zeros(1, 4);
        try
            [detections, scores] = detect(img, w, hogCellSize, scales);
            keep = boxsuppress(detections, scores, 0.25);
            detections = detections(:, keep(1:10))';
            % disp(detections);
            detections(:, 3:end) = detections(:, 3:end) - detections(:, 1:2);
            scores = scores(keep(1:10));
        catch
            % Pass
        end
        results = [detections scores(:)];
        txtFileName = fullfile(imPath{1}, [img_names(f).name, '.txt']);
        fh = fopen(txtFileName, 'w');
        fprintf(fh, '%s\n', img_names(f).name);
        fprintf(fh, '%d\n', numel(scores));
        fprintf(fh, '%d %d %d %d %g\n', results);
        fclose(fh);
    end
end
