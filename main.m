clc
close all

%-------Task 1: Region of Interest-------
v=VideoReader('april21.avi'); 
count1=1;
detector1 = zeros(480,704,3,300);
while hasFrame(v)
    frame = readFrame(v);

    %Find horizontal edges of grayscale image 
    gradient = edge(rgb2gray(frame),'sobel','horizontal');
    gray = gradient*255;
    %Calculate the maximum variance between all rows in each frame
    variance_rows= sum(gray,2);
    if count1==1
        [M,I] = max(variance_rows);
        I=I-10;%for correction purposes
    end
    %Limit frame to ROI
    frame(1:I, 1:704, :) = 0;
    
    assist = zeros(I,704,3);
    detector1(:,:,:,count1) = [assist;frame(I+1:480,:,:)];
    count1=count1+1;
end

vw1 = VideoWriter('roi.avi');
open(vw1);
writeVideo(vw1,uint8(detector1));
close(vw1)

%-------Task 2: Add Noise and then denoise Video-------
v=VideoReader('april21.avi');
count2=1;
detector2 = zeros(480,704,3,300);
while hasFrame(v)
    frame2 = readFrame(v);
    
    %Add Gaussian and Salt & Pepper Noise
    frame_gaussian = imnoise(frame2,'gaussian');
    frame_gaussian_sp = imnoise(frame_gaussian,'salt & pepper');
    
    %Median filter first, Average filter second
    frame_sp_filtered_red=medfilt2(frame_gaussian_sp(:,:,1),[3 3]);
    frame_sp_filtered_green=medfilt2(frame_gaussian_sp(:,:,2),[3 3]);
    frame_sp_filtered_blue=medfilt2(frame_gaussian_sp(:,:,3),[3 3]);
    
    avg=fspecial('average',3);
    frame_filtered_red=imfilter(frame_sp_filtered_red,avg);
    frame_filtered_green=imfilter(frame_sp_filtered_green,avg);
    frame_filtered_blue=imfilter(frame_sp_filtered_blue,avg);
    
    frame_filtered=cat(3,frame_filtered_red,frame_filtered_green,frame_filtered_blue);
   
    %Find horizontal edges of grayscale image 
    gradient2 = edge(rgb2gray(frame_filtered),'sobel','horizontal');
    gray2 = gradient2*255;
    %Make top and bottom sides black because 704 pixels width with 3x3 masks do not
    %divide
    gray2(1:2,:) = 0;
    gray2(479:480,:)=0;
    %Calculate the maximum variance between all rows in each frame
    variance_rows2= sum(gray2,2);
    if count2==1
        [M2,I2] = max(variance_rows2);
    end
    %Limit frame to ROI
    frame_filtered(1:I2, 1:704, :) = 0;
    
    assist2 = zeros(I2,704,3);
    detector2(:, :, :, count2) = [assist2;frame_filtered(I2+1:480,:,:)];
    
    count2 = count2 + 1;
end

vw2 = VideoWriter('denoise.avi');
open(vw2);
writeVideo(vw2,uint8(detector2));
close(vw2)

%-------Task 3: Masks to find the cars-------
v=VideoReader('april21.avi');
count3=1;
max_sat = 0;
detector3 = zeros(480,704,1,300);
while hasFrame(v)
    frame3 = readFrame(v);
    
    %Find horizontal edges of grayscale image 
    gradient3 = edge(rgb2gray(frame3),'sobel','horizontal');
    gray3 = gradient3*255;
    %Calculate the maximum variance between all rows in each frame
    variance_rows3= sum(gray3,2);
    if count3==1
        [M3,I3] = max(variance_rows3);
        I3=I3-5;%for correction purposes
    end
    %Limit frame to ROI
    frame3(1:I3, 1:704, :) = 0;

    %Color segmentation to highlight lanes
    %Turn to YCbCr color space
    ycbcr = zeros(480-I3,704,3);
    ycbcr = rgb2ycbcr(frame3(I3:480,:,:));
    
    %Extract color features
    col_seg = zeros(480-I3,704);
    Uy = mean2(ycbcr(:,:,1));
    Sy = std2(ycbcr(:,:,3));
    Ty = Uy + 5.5*Sy;
    
    Ucr = mean2(ycbcr(:,:,1));
    Scr = std2(ycbcr(:,:,3));
    Tcr = Ucr + 5.5*Scr;
    
    for j=1:(480-I3)
        for k=1:704
            if ycbcr(j,k,1) > Ty && ycbcr(j,k,3) > Tcr
                col_seg(j,k) = 1;
            else
                col_seg(j,k) = 0;
            end
        end
    end

    %Hough Transform to concentrate on cars between the lanes
    a = col_seg;
    [H,T,R] = hough(a);

    P  = houghpeaks(H,1,'threshold',ceil(0.3*max(H(:))));
    x = T(P(:,2)); y = R(P(:,1));

    lines = houghlines(a,T,R,P,'FillGap',5,'MinLength',7);
    ang = (lines(1).point2(2)-lines(1).point1(2))/(lines(1).point2(1)-lines(1).point1(1));
    beta = lines(1).point2(2)-ang*lines(1).point2(1);

    %Turn to HSV color space
    hsv_frame=rgb2hsv(frame3(I3+1:480,1:704,:));
    
    for i = 1:(480-I3)
        for j =1:704
            if (i-ang*j-beta)<7
                hsv_frame(i,j,2)=0;
            end
        end
    end
    
    %Find asphalt and generally areas with low saturation and erase them to
    %concentrate on cars
    if count3==1
        max_sat = max(max(hsv_frame(:,:,2)));
    end
    
    for i = 1:(480-I3)
        for j =1:704
            if hsv_frame(i,j,2)<(max_sat)
                hsv_frame(i,j,2)=0;
            end
        end
    end
    
    %To filter out all small parts that do not belong to the cars
    proc_frame = bwareafilt(logical(hsv_frame(:, :, 2)),[20 5000]);
    
    assist3 = zeros(I3,704,1);
    %Canny ege detector to get the edges of the vehicles
    detector3(:, :, 1, count3) = [assist3;edge(proc_frame,'canny')];
    
    count3 = count3 + 1;
end

vw3 = VideoWriter('gradient.avi');
open(vw3);
writeVideo(vw3,detector3);
close(vw3)


