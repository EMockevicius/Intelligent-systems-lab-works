function hsv_value=spalva_color(Im)

BW = im2bw(rgb2gray(Im),0.95);
BW = imfill(~BW,'holes');
BW = imopen(BW,strel('disk',12));


hsv_im=rgb2hsv(Im);
hsv=hsv_im(:,:,1);

hsv_value=mean(mean(hsv(BW)));
