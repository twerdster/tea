
function genDataAll(dataDir, outDir, nFeatures, probeSize, featurePrecision, rseed, sampleRate, createTeaMulti, createTeaSolo, createLibSVM)
%Eg usage: genTempData('/home/gipadmin/Downloads/TestData/','/home/gipadmin/forks/tea/data/',200,50,'int8',200,0)
isO = exist('OCTAVE_VERSION', 'builtin') ~= 0;
warning off;
more off;
fclose all;

% Harcoded:
maxLabel = 7;
maxDiff = 250; %25cm

dirInfo=dir([dataDir '/Data_*.mat']);
fprintf('Found %i files\nScanning...\n',length(dirInfo));
nImages = length(dirInfo);
totSamples = 0;

if nImages == 0
    dataDir
    error('No images available');
    return;
end
%h = waitbar(0,'Calculating total number of samples ...');

for i=1:100:nImages
    load([dataDir dirInfo(i).name]);
    %depth = Depth;
    totSamples = totSamples + sampleRate*sum(lbl(:)>=1);
    if mod(i,1000)==0
        fprintf('Scanned %i / %i images\n',i,nImages);
        if isO
            fflush(stdout);
        else
            drawnow;
        end
    end
    %waitbar(i/nImages,h);
end
totSamples = totSamples*100;

fprintf('TotalSamples : %i\n\n',totSamples);
%delete(h);

ISINT8 =        strcmp(lower(featurePrecision),'int8');
ISINT16 =       strcmp(lower(featurePrecision),'int16');
ISFLOAT32 =     strcmp(lower(featurePrecision),'single');

if ~ISINT8 & ~ISINT16 & ~ISFLOAT32
    error('Unsupported bit depth.')
end

if ISINT8       featureMax = 127; end
if ISINT16      featureMax = 32767; end
if ISFLOAT32    featureMax = 127; end

rand('seed',rseed);

rot = @(P,dA)  [ cos(dA) -sin(dA); sin(dA) cos(dA)] * P;
expp=@(s,n) (exp(-s:s/(n-1):0)-exp(-s))/(1-exp(-s));

nDiv = 20;
steepness = 1.3;
radius = probeSize*(0:1/(nDiv):1);
radius = probeSize*expp(steepness,nDiv+1);
radius=radius/2;
radius = radius(randperm(nDiv)+1);
numRots = nFeatures/nDiv;
delta = 2*pi/numRots;


coords_ = [];
for i=1:nDiv
    dA = delta * (i-1)/double(nDiv) + rand(1)*delta*0.5;
    coords_ = [coords_ rot([0 radius(i)]',dA)];
end

coords = [];
for i=1:numRots
    dA = delta*i;
    coords = [coords rot(coords_,dA)];
end

coords = [coords; (rand(size(coords))-0.5)*0.1*probeSize]; % the 0.1 here makes feature roots closer to the center of the patch

fprintf('Number of output features: %i\n',size(coords,2));

fBase = 'F_';
mkdir(outDir);
coordsFile         = [outDir 'Coords.co'];
teaLabelFile       = [outDir 'Labels.lbl'];
threshFile         = [outDir 'Threshholds.thr'];
lsvmFile           = [outDir 'libSVM.out'];
teaBinaryDataFile  = [outDir 'teaData.out'];
teaBinaryLabelFile = [outDir 'teaLabel.out'];

% remove all data files
delete(coordsFile);
delete(teaLabelFile);
delete(threshFile);
delete(lsvmFile);
delete(teaBinaryDataFile);
delete(teaBinaryLabelFile);
delete([outDir '*.feat']);

fLabel     = fopen(teaLabelFile,'wb');
fTeaData   = fopen(teaBinaryDataFile,'wb');
fTeaLabel  = fopen(teaBinaryLabelFile,'wb');
flsvm      = fopen(lsvmFile,'wt');

featSize=totSamples*sizeof(featurePrecision);
soloFeatSize = featSize*nFeatures;

%system(['head -c ' num2str(soloFeatSize) ' </dev/zero >' teaBinaryDataFile]);
      
fMin = [];
fMax = [];
imCnt=0;
fileHandles = [];
if createTeaMulti
for feat = 1:nFeatures
    featFile = [outDir fBase num2str(feat-1,'%.4i') '.feat'];
%    system(['head -c ' num2str(featSize) ' </dev/zero >' featFile]);
    fileHandles(feat) = fopen(featFile,'wb');
    fMin(feat) = featureMax;
    fMax(feat) = -featureMax;
end
end
tic
classInds = 0:nFeatures-1;

for imageIndex = 1:nImages
    % Load the depth and label images
    A=load([dataDir dirInfo(imageIndex).name]);
    depth = A.depth;
    label = A.lbl;
    depth = single(depth);
    
    % Sample all values that are part of the hand
    inds = int32(find(label>=1));
    
    %Turn the background into -1 (actually it will be zero because its uint8) and the first label into 0
    % but its fine because the bg labels are ignored 
    label = label-1;
    nSampled = length(inds);
    nSamples = ceil(double(sampleRate*nSampled));
    
    % And they need to be randomly sampled so theyre not all at the
    % start of each image
    perm = int32(randperm(nSampled));
    inds = inds(perm(1:nSamples));
    [i,j]=ind2sub(size(label),inds);
    coords_ = single(coords);
    
    %needs to be compiled with mex getDepthFeatures.cpp COMPFLAGS="/openmp $COMPFLAGS"
    data    = getDepthFeatures(int32(i), int32(j), single(depth), single(coords_), single(maxDiff), single(featureMax));
    labels  = label(inds);
    
    for feat=1:nFeatures
        fMin(feat) = min(fMin(feat),min(data(:,feat)));
        fMax(feat) = max(fMax(feat),max(data(:,feat)));
    end    
    
    %% Write features for Tea
    %write data and labels to feature files
    if createTeaMulti
    for feat=1:nFeatures
        fwrite(fileHandles(feat),data(:,feat),featurePrecision);
    end
    fwrite(fLabel,labels,'uint16');
    end
    
     %% Write features for binary Tea files
    if createTeaSolo
    fwrite(fTeaData,data',featurePrecision);
    fwrite(fTeaLabel,labels,'uint16');
    end
    
    %% Write features for libSVM
     %write data and labels to data
     if createLibSVM
     if ISFLOAT32
    for r=1:size(data,1)
        fprintf(flsvm,'%i',labels(r));
        fprintf(flsvm,' %i:%f', [classInds; data(r,:)]);
        fprintf(flsvm,'\n');
    end
    else
    for r=1:size(data,1)
        fprintf(flsvm,'%i',labels(r));
        fprintf(flsvm,' %i:%i', [classInds; data(r,:)]);
        fprintf(flsvm,'\n');
    end
    end
    end
    
   
    
    if mod(imageIndex,50) == 0
        toc
        tic;
        fprintf('Image %i / %i\n ',imageIndex,nImages);
        if isO
            fflush(stdout);
        else
            drawnow;
        end
        
        imshow(depth,[]);
        hold on;
        plot(j,i,'r.','MarkerSize',1);
        hold off;
        drawnow;
    end
    
end

if createTeaMulti
for feat = 1:nFeatures
    fclose(fileHandles(feat));
end
end

fclose(fLabel);
writeDataThresholds(outDir,nFeatures,featurePrecision,featureMax,fBase,threshFile,fMin,fMax);

fCoords=fopen([outDir 'Coords.co'],'wb');
fwrite(fCoords,coords,'single');
fclose(fCoords);
fclose all;
end


function writeDataThresholds(outDir,numF,featurePrecision,maxVal, fBase, threshFile,fMin,fMax)

tFile=fopen(threshFile,'wb');
fprintf('Writing thresholds ...\n');

for feat=0:numF-1
    fprintf('%i ... ',feat);
    featureMin = fMin(feat+1);%single(min(data(data~=fMin)));       % Minimum feature value
    featureMax = fMax(feat+1);%single(max(data(data~=fMax)));       % Maximum feature value
    
    fwrite(tFile,maxVal,'single');
    fwrite(tFile,featureMin,'single');
    fwrite(tFile,featureMax,'single');
    
    fprintf(' Done.\n');
end

fclose(tFile);

end


function nbytes = sizeof(precision)
try
    z = zeros(1, precision); %#ok, we use 'z' by name later.
catch
    error('Unsupported class for finding size');
end
w = whos('z');
nbytes = w.bytes;
end
