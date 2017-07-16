
function genData(dataDir, outDir, nFeatures, probeSize, featurePrecision, maxFilesOpen, rseed)
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

for i=1:nImages
    load([dataDir dirInfo(i).name]);
    %depth = Depth;
    totSamples = totSamples + sum(lbl(:)>=1);
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

fprintf('TotalSamples : %i\n\n',totSamples);
%delete(h);

ISINT8 =        strcmp(lower(featurePrecision),'int8');
ISINT16 =       strcmp(lower(featurePrecision),'int16');
ISINT32 =       strcmp(lower(featurePrecision),'int32');
ISFLOAT32 =     strcmp(lower(featurePrecision),'single');

if ~ISINT8 & ~ISINT16 & ~ISINT32 & ~ISFLOAT32
    error('Unsupported bit depth.')
end

if ISINT8       featureMax = 127; end
if ISINT16      featureMax = 32767; end
if ISINT32      featureMax = 2147483500; end
if ISFLOAT32    featureMax = 3.3e+037; end


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
bot = 1;
mkdir(outDir);
coordsFile = [outDir 'Coords.co'];
labelFile = [outDir 'Labels.lbl'];
threshFile = [outDir 'Threshholds.thr'];
posesFile = [outDir 'Poses.pose'];


featSize = totSamples*sizeof(featurePrecision);

% remove all data files
delete(coordsFile);
delete(labelFile);
delete(threshFile);
delete(posesFile);
delete([outDir '*.feat']);

coordsSize = (4*nFeatures)*sizeof('single');
labelSize = totSamples*sizeof('int16');
posesSize = totSamples*sizeof('int32');

system(['head -c ' num2str(coordsSize) ' </dev/zero >' coordsFile]);
system(['head -c ' num2str(labelSize) ' </dev/zero >' labelFile]);
system(['head -c ' num2str(posesSize) ' </dev/zero >' posesFile]);

fLabel=fopen(labelFile,'r+b');
fPose=fopen(posesFile,'r+b');

fMin = [];
fMax = [];
while 1
    top = min(bot+maxFilesOpen-1,nFeatures);
    fprintf('\nGenerating features %i to %i ...\n',bot-1,top-1);
    if isO
        fflush(stdout);
    else
        drawnow;
    end
    
    Q = bot:top;
    imCnt=0;
    fileHandles = [];
    for feat = Q
        featFile = [outDir fBase num2str(feat-1,'%.4i') '.feat'];
        system(['head -c ' num2str(featSize) ' </dev/zero >' featFile]);
        fileHandles(feat) = fopen(featFile,'r+b');
        fMin(feat) = featureMax;
        fMax(feat) = -featureMax;
    end
    tic
    for imageIndex = 1:nImages
        imCnt = imCnt+1;
        
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
        
        % We need to process EXACTLY the same data points for each feature
        % batch that we process
        rand('seed',imageIndex + rseed);
        % And they need to be randomly sampled so theyre not all at the
        % start of each image
        perm = int32(randperm(nSampled));
        inds = inds(perm);%(1:nSamples));
        [i,j]=ind2sub(size(label),inds);
        coords_ = single(coords(:,Q));
        
        %needs to be compiled with mex getDepthFeatures.cpp COMPFLAGS="/openmp $COMPFLAGS"
        data = getDepthFeatures(int32(i), int32(j), single(depth), single(coords_), single(maxDiff), single(featureMax));
        
        labels = label(inds);
        poses = ones(size(data,1),1)*(imageIndex-1);
        
        %write data and labels to feature files
        for feat=Q
            f = feat-Q(1)+1;
            fwrite(fileHandles(feat),data(:,f),featurePrecision);
            ind = abs(data(:,f))~=featureMax;
            if all(~ind)
                continue;
            end
            fMin(feat) = min(fMin(feat),min(data(ind,f)));
            fMax(feat) = max(fMax(feat),max(data(ind,f)));
        end
        if Q(1) == 1 % We only need to run this once for the outer iteration
            fwrite(fLabel,labels,'uint16');
            fwrite(fPose, poses, 'uint32');
        end
        if mod(imCnt,50) == 0
            toc
            tic;
            fprintf('Image %i / %i\n ',imageIndex,nImages);
            if isO
                fflush(stdout);
            else
                drawnow;
            end
            
            %imshow(depth,[]);
            %hold on;
            %plot(j,i,'r.','MarkerSize',1);
            %hold off;
            %drawnow;
        end
        
    end
    for feat = Q
        fclose(fileHandles(feat));
    end
    
    %This is our end condition
    if Q(end) == nFeatures
        break;
    end
    bot = top + 1;
end

fclose(fLabel);
fclose(fPose);
%Write thresholds
writeDataThresholds(outDir,nFeatures,featurePrecision,featureMax,fBase,threshFile,fMin,fMax);

fCoords=fopen([outDir 'Coords.co'],'r+b');
fwrite(fCoords,coords,'single');
fclose(fCoords);
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
