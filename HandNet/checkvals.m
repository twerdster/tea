
f=fopen('/home/twerd/forks/tea/data/Train_3k/F_0000.feat','rb');
f0 = fread(f,inf,'int8');

f=fopen('/home/twerd/forks/tea/data/Train_3k/F_0099.feat','rb');
f99 = fread(f,inf,'int8');

f=fopen('/home/twerd/forks/tea/data/Train_3k/teaData.out','rb');
td = fread(f,inf,'int8');
td=td(1:length(f99)*100);
td=reshape(td,100,length(f99));

f=fopen('/home/twerd/forks/tea/data/Train_3k/teaLabel.out','rb');
tl = fread(f,inf,'uint16');

f=fopen('/home/twerd/forks/tea/data/Train_3k/Labels.lbl','rb');
l = fread(f,inf,'uint16');

fid=fopen('/home/twerd/forks/tea/data/Train_3k/libSVM.out','rt');
i=1;
while ~feof(fid) % not end of the file 
       s = fgetl(fid); % get a line 
       ad=sscanf(strrep(s,':',' '),'%f')';
       al=ad(1);
       ad=ad(3:2:end);if mod(i,1000)==0
       fprintf('%i %i %i - ',i,sum(al~=l(i)),sum(td(:,i)'~=ad));
       fflush(stdout);
       end
       i=i+1;
end
yy(1,:)=[];
y=yy;



sum(f99'~=td(end,:))
sum(f0'~=td(1,:))
sum(tl~=l)