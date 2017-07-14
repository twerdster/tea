function expandData(inDataDir, inDataType, outDataDir, outNumSamples, outNumFeatures, outNumClasses, outDataType)
%% Detect input data
d=dir([inDataDir '*.feat']);

inNumFeatures = length(d);

f=fopen([inDataDir 'F_0000.feat'],'rb');
a=fread(f,inf,[inDataType]);
inNumSamples=length(a);
fclose(f);

f=fopen([inDataDir 'Labels.lbl'],'rb');
a=fread(f,inf,'uint8');
inNumClasses=max(a)+1;
fclose(f);

fprintf('Detected: inSamples=%gm, inFeatures=%i, inClasses=%i\n',uint32(inNumSamples/1e6),inNumFeatures,inNumClasses);
if ~exist('outNumSamples')
    return;
end
%% Write new expanded data
ind = mod(0:(outNumSamples-1),inNumSamples)+1;
mkdir(outDataDir);

for i = 0:outNumFeatures-1
    f=fopen(sprintf('%sF_%04i.feat',inDataDir,mod(i,inNumFeatures)),'rb');
    a=fread(f,inf,[inDataType '=>' outDataType]);
    fclose(f);
    f=fopen(sprintf('%sF_%04i.feat',outDataDir,i),'wb');
    a=a(ind);
    fwrite(f,a,outDataType);
    fclose(f);
    fprintf('Completed: %sF_%04i.feat\n',outDataDir,i)
end

f=fopen(sprintf('%sLabels.lbl',inDataDir),'rb');
a=fread(f,inf, 'uint8=>uint8' );
fclose(f);
f=fopen(sprintf('%sLabels.lbl',outDataDir),'wb');
divs = uint8(ceil(outNumClasses/inNumClasses));
a=a(ind)*divs + uint8(randi(divs,length(ind),1))-1;
a=min(a,outNumClasses-1);
fwrite(f,a,'uint8');
fclose(f);


f=fopen(sprintf('%sThreshholds.thr',inDataDir),'rb');
a=fread(f,inf, 'single=>single' );
fclose(f);
f=fopen(sprintf('%sThreshholds.thr',outDataDir),'wb');
a=a(mod((1:outNumFeatures*3)-1,inNumFeatures*3)+1);
fwrite(f,a,'single');
fclose(f);

f=fopen(sprintf('%sCoords.co',inDataDir),'rb');
a=fread(f,inf, 'single=>single' );
fclose(f);
f=fopen(sprintf('%sCoords.co',outDataDir),'wb');
a=a(mod((1:outNumFeatures*4)-1,inNumFeatures*4)+1);
fwrite(f,a,'single');
fclose(f);


%% Detect resulting output data
d=dir([outDataDir '*.feat']);

outNumFeatures = length(d);

f=fopen([outDataDir 'F_0000.feat'],'rb');
a=fread(f,inf,[outDataType]);
outNumSamples=length(a);
fclose(f);

f=fopen([outDataDir 'Labels.lbl'],'rb');
a=fread(f,inf,'uint8');
outNumClasses=max(a)+1;
fclose(f);

fprintf('Result: outSamples=%gm, outFeatures=%i, outClasses=%i\n',uint32(outNumSamples/1e6),outNumFeatures,outNumClasses);


