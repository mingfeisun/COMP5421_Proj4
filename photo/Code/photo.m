%% main
%*****************************************main*******************************************************
function result = photo()
  close all;
  clear all;
  % declare
  absolutePath = 'data/data08/';
  lightVecFile = 'lightvec.txt';

  % light vect load
  dataFile = fopen(strcat(absolutePath, lightVecFile), 'r');
  formatSpec = '%f %f %f';
  sizeData = [3 Inf];
  data = fscanf(dataFile, formatSpec, sizeData );
  fclose(dataFile);
  data = data';
  x = data(:,1); y = data(:,2); z = data(:,3); 

  % process
  surfaceNormal = initialNormal(x, y, z, absolutePath );
  figure();imshow(surfaceNormal);title('Initial Normal Estimation');
  figure();recsurf = shapeFromShapelets(surfaceNormal );title('Initial Surface');
  newSurfaceNormal = graphCutSurfaceNormal( surfaceNormal );
  figure();imshow(newSurfaceNormal);title('Refined Normal Estimation');
  figure();newRecsurf = shapeFromShapelets(newSurfaceNormal );title('Refined Surface');
  result = 1;
end

%% initialNormal
%****************************************************************************************************
function surfaceNormal = initialNormal(x, y, z, absolutePath)
  [imgIntensity, lights, denominatorLight] = preProcessData(x, y, z, absolutePath); % imgIntensity here alread divide Denominator image
  [M, N, s] = size(imgIntensity);
  matrixToFindNormal = zeros(s, 3);
  surfaceNormal = zeros(M, N, 3);
  tic;
  for i = 1:M
      for j = 1:N
          for k = 1:s
             matrixToFindNormal(k, :) = [lights(k,1)-imgIntensity(i,j,k)*denominatorLight(1),...
                                          lights(k,2)-imgIntensity(i,j,k)*denominatorLight(2),...
                                           lights(k,3)-imgIntensity(i,j,k)*denominatorLight(3)];
          end
          [~,~,v]=svd(matrixToFindNormal);
          temp = v(:,end);
          if temp(3)<0
              temp = -temp;
          end
          surfaceNormal(i,j,:)=temp;
      end
  end
  toc;
end

%% preProcessData
%****************************************************************************************************
function [imageMatrices, lightDirections, denoLight] = preProcessData(x, y, z, absolutePath)
  [~,~,index] = resamplingData(x, y, z);
  s = size(index, 1);
  [width, height, ~] = size(imread(strcat(absolutePath, 'image0001.bmp')));
  imageMatrices = zeros(width , height , s); 
  lightDirections = zeros(s, 3);
  for i = 1:s
     lightDirections(i, :) = [x(index(i)), y(index(i)), z(index(i))];

     if index(i)<10
         path = strcat(absolutePath, 'image000', num2str(index(i)), '.bmp');
     elseif index(i)<100
         path = strcat(absolutePath, 'image00', num2str(index(i)), '.bmp');
     elseif index(i)<1000
         path = strcat(absolutePath, 'image0', num2str(index(i)), '.bmp');
     else
         path = strcat(absolutePath, 'image', num2str(index(i)), '.bmp');
     end

     img = imread(path);
     img = rgb2gray(img);
     imageMatrices(:, :, i) = double(img);
  end

  % find denominator image
  percentile = 0.9;
  intensitySum = sum(sum(imageMatrices,1),2);
  [~,idx] = sort(intensitySum);
  denominatorImage = imageMatrices(:,:,idx(floor(percentile*s)));
  figure();
  imshow(uint8(denominatorImage));
  title('Denominator Image');

  for i = 1:s
      % divide denominator image
      imageMatrices(:,:,i) = imageMatrices(:,:,i)./denominatorImage;
  end
  imageMatrices(:, :, idx(floor(percentile*s))) = [];
  denoLight = lightDirections(idx(floor(percentile*s)), :);
  lightDirections(idx(floor(percentile*s)), :) = [];
end

%% resamplingData
%****************************************************************************************************
function [result, index, uniqueIndex] = resamplingData(x, y, z)
  subSamples = icosahedron(0.2); % subdivided icosahedron
  %scatter3(subSamples(:,1), subSamples(:,2), subSamples(:,3));
  s = size(subSamples, 1);
  result = [];
  index = [];
  for i = 1:s
      d = (x-subSamples(i, 1)).^2 + (y-subSamples(i, 2)).^2 + (z-subSamples(i, 3)).^2;
      [~, order] = sort(d);
      result = [result; [x(order(1)), y(order(1)), z(order(1))]];
      index = [index; order(1)];
  end
  %scatter3(result(:,1), result(:,2), result(:,3));
  uniqueIndex = unique(index); %delete duplicate
end

%% shapeFromShapelets 
%****************************************************************************************************
function recsurf = shapeFromShapelets(surfaceNormal)
  [M, N, ~] = size(surfaceNormal);
  slant = zeros(M, N);
  tilt = zeros(M, N);
  for i = 1:M
      for j = 1:N
          x = surfaceNormal(i, j ,1);
          y = surfaceNormal(i, j ,2);
          z = surfaceNormal(i, j ,3);
          slant(i, j) = x;%-atan(sqrt(x^2+y^2)/z);
          tilt(i, j)  = y;%acos(x/sqrt(x^2+y^2))+pi*(y<0);
      end
  end
  recsurf = shapeletsurf(slant, tilt, 6, 3, 2);
  surf(recsurf, 'FaceColor', 'red', 'EdgeColor', 'none');
  camlight left; lighting phong;
end

%% graphCutSurfaceNormal 
%****************************************************************************************************
function  newSurfaceNormal = graphCutSurfaceNormal(surfaceNormal)
  tic;

  lambda = 0.01;
  sigma = 1;
  epsilon = 0.01;

  [W, H, ~] = size(surfaceNormal);
  Icosahedron = icosahedron(0.1);
  [sIco, ~] = size(Icosahedron);

  NormalLabel = zeros(W, H);
  for i = 1:W
      for j = 1:H
          d = (Icosahedron(:,1)-surfaceNormal(i,j,1)).^2 + ...
              (Icosahedron(:,2)-surfaceNormal(i,j,2)).^2 + ...
              (Icosahedron(:,3)-surfaceNormal(i,j,3)).^2;
          [~, index] = min(d);
          NormalLabel(i, j) = index;
      end
  end

  WH = W*H;
  segclass = reshape(NormalLabel, 1, [])';  % initial Normal
  pairwise = sparse(WH,WH);                 % neighborhood term 
  unary = zeros(sIco,WH);                   % cost connect to each label
  labelcost = zeros(sIco, sIco);            % smoothness term

  for row = 0:H-1
    for col = 0:W-1
      pixel = 1+ row*W + col;
      if row+1 < H 
          Na = segclass(pixel);
          Nb = segclass(1+col+(row+1)*W);
          ico = Icosahedron(Na, :) - Icosahedron(Nb, :);
          ico = sqrt(ico*ico');
          pairwise(pixel, 1+col+(row+1)*W) = lambda * log(1 + ico/sigma);
      end
      
      if row-1 >= 0 
          Na = segclass(pixel);
          Nb = segclass(1+col+(row-1)*W);
          ico = Icosahedron(Na, :) - Icosahedron(Nb, :);
          ico = sqrt(ico*ico');
          pairwise(pixel, 1+col+(row-1)*W) = lambda * log(1 + ico/sigma); 
      end
      
      if col+1 < W 
          Na = segclass(pixel);
          Nb = segclass(1+(col+1)+row*W);
          ico = Icosahedron(Na, :) - Icosahedron(Nb, :);
          ico = sqrt(ico*ico');
          pairwise(pixel, 1+(col+1)+row*W) = lambda * log(1 + ico/sigma); 
      end
      
      if col-1 >= 0 
          Na = segclass(pixel);
          Nb = segclass(1+(col-1)+row*W);
          ico = Icosahedron(Na, :) - Icosahedron(Nb, :);
          ico = sqrt(ico*ico');
          pairwise(pixel, 1+(col-1)+row*W) = lambda * log(1 + ico/sigma); 
      end 
      
      for i = 1:sIco
          ico = Icosahedron(i, :) - Icosahedron(segclass(pixel), :);
          unary(i,pixel) = sqrt(ico*ico');
      end
    end
  end

  for i = 1:sIco
      for j = 1:sIco
          Sij = Icosahedron(i, :) - Icosahedron(j, :);
          Sij = sqrt(Sij*Sij');
          K = 1 + epsilon - exp(-(2-Sij)/sigma^2);
          labelcost(i, j) = lambda*K*Sij;
      end
  end


  toc;

  tic;
  segclass = segclass - 1;
  [labels, ~, ~] = GCMex(segclass, single(unary), pairwise, single(labelcost), 1);
  labels = labels + 1;
  toc;

  labels = reshape(labels, W, H);
  newSurfaceNormal = zeros(W, H, 3);

  for i = 1:W
      for j = 1:H
          newSurfaceNormal(i, j, :) = Icosahedron(labels(i,j), :);
      end
  end
end
