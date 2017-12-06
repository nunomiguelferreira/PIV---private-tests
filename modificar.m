load('CalibrationData.mat');


i=(1:148)';
imgseq1.img=num2str(i,'rgb_image1_%d.png')
imgseq1.depth=num2str(i,'depth1_%d.mat')
imgseq2.img=num2str(i,'rgb_image2_%d.png')
imgseq2.depth=num2str(i,'depth2_%d.mat')

    for i = 1:148;
        rgbData(:,:,:,i) = imread(strtrim(imgseq1.img(i,:))); 
        depthData(:,i) = load(strtrim(imgseq1.depth(i,:)));
        rgbData1(:,:,:,i) = imread(strtrim(imgseq2.img(i,:))); 
        depthData1(:,i) = load(strtrim(imgseq2.depth(i,:)));
    end;

%% 
figure
%for i = 1:148;
i = 1;
    xyz=get_xyzasus(depthData(i).depth_array(:),[480 640],1:640*480,Depth_cam.K,1,0);
    im = rgbData(:,:,:,i);
    %Compute "virtual image" aligned with depth
    rgbd=get_rgbd(xyz,im,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);
    imagesc([im; rgbd])
    cl=reshape(rgbd,480*640,3);
    p1=pointCloud(xyz,'Color',cl);
    % Point cloud with colour per pixel
    showPointCloud(p1); %% point cloud da primeira cam, first frame 
    
    figure %% pointcloud da segunda cam, first frame
    
    xyz=get_xyzasus(depthData1(i).depth_array(:),[480 640],1:640*480,Depth_cam.K,1,0);
    im1 = rgbData1(:,:,:,i);
    %Compute "virtual image" aligned with depth
    rgbd=get_rgbd(xyz,im,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);
    imagesc([im; rgbd])
    cl=reshape(rgbd,480*640,3);
    p2=pointCloud(xyz,'Color',cl);
    % Point cloud with colour per pixel
    showPointCloud(p2);    
    
    %%
    
m1=depthData(1).depth_array(:)>0;
m2=depthData1(1).depth_array(:)>0;
 
imaux1=double(repmat(m1,[1,1,3])).*double(im1)/255;
imaux2=double(repmat(m2,[1,1,3])).*double(im2)/255;

figure(5);
imagesc(imaux1);
figure(6);
imagesc(imaux2);
%%
% select figure with im1 and click 5 points
%[u1,v1]=ginput(5);
%select figure with im2 and click in the corresponding points
%[u2 v2]=ginput(5);
u1 =[  216.7007  324.5143 329.1022 352.0412 372.6864]';
v1 =[  53.7925 80.2550 111.8629 194.9257 78.7848]';
u2=[257.9910  224.7294  135.2670   42.3638  192.6147]';
v2=[87.6057  108.1876  124.3591  168.4632  130.9747]';

figure(5);hold on;plot(u1,v1,'*r');hold off;
figure(6);hold on;plot(u2,v2,'*r');hold off;
ind1=sub2ind([480 640],uint64(v1),uint64(u1));
ind2=sub2ind([480 640],uint64(v2),uint64(u2));
%% Compute Centroids
cent1=mean(xyz1(ind1,:))';
cent2=mean(xyz1(ind2,:))';
pc1=xyz1(ind1,:)'-repmat(cent1,1,5);
pc2=xyz2(ind2,:)'-repmat(cent2,1,5);
[a b c]=svd(pc2*pc1')
R12=a*c'
%%
xyzt1=R12*(xyz1'-repmat(cent1,1,length(xyz1)));
xyzt2=xyz2'-repmat(cent2,1,length(xyz2));
T=cent2-R12*cent1;
%%
ptotal=pointCloud([xyzt1';xyzt2'],'Color',[cl1;cl2]);
figure(7);
showPointCloud(ptotal);
 

%end
%

  


%%

se = strel('square',3);

foregroundDetector = vision.ForegroundDetector('NumGaussians', 2, 'NumTrainingFrames', 2, 'LearningRate', 0.007);

for k = 1:148
	% Create an image filename, and read it in to a variable called imageData.
		imageData = rgbData(:,:,:,k);
        frame = rgb2gray(imageData); % read the next video frame
        foreground = step(foregroundDetector, frame);
        filteredForeground = imopen(foreground,se);
        blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
        'AreaOutputPort', false, 'CentroidOutputPort', false, ...
        'MinimumBlobArea', 1250);
        bbox = step(blobAnalysis, filteredForeground); %bbox stores the coordinates in x and y pixels of the detected object
        A = isempty(bbox);
        if (A == 0)
            temp(k).bbox = bbox; %Xmin,Xmax,Ymin,Ymax dos objectos tracked
            temp(k).numobj = size(bbox,1); %numero de objectos detectados nesse frame.
        end;
        result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');
        numobj = size(bbox, 1);
        %result = insertText(result, [10 10], numobj, 'BoxOpacity', 1, ...
       % 'FontSize', 14);
        title('Detected Objects');
        imshow(result);
end