
%thresholds t3
%------------
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/200-100-7-char/'],100*1e6,200,7,'int8');

%dataypes t10
%------------
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/100-100-7-CHAR/'],100*1e6,100,7,'int8');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/100-100-7-SHORT/'],100*1e6,100,7,'int16');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/100-100-7-FLOAT/'],100*1e6,100,7,'single');

%distributions  t4
%------------
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-7-unbalanced/'],100*1e6,20,7,'int8','unbalanced');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-7-uniform/'],100*1e6,20,7,'int8','uniform');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-7-real/'],100*1e6,20,7,'int8','real');

%folding t5
%------------
%expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-7-char/'],100*1e6,20,7,'int8');

%depths t6
%------------
%expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/200-100-7-char/'],100*1e6,200,7,'int8');

%gpus t7
%------------
%expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/200-100-7-char/'],100*1e6,200,7,'int8');

%preloaded t8
%------------
%expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/200-54-7-char/'],54*1e6,200,7,'int8');

%numclasses t11
%------------
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-2-char/'],100*1e6,20,2,'int8');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-7-char/'],100*1e6,20,7,'int8');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-32-char/'],100*1e6,20,32,'int8');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-64-char/'],100*1e6,20,64,'int8');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-65-char/'],100*1e6,20,65,'int8');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-256-char/'],100*1e6,20,256,'int8');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-257-char/'],100*1e6,20,257,'int8');
expandData([baseDir '/data/200-54-7-char/'],'int8',[outDir '/data/20-100-1000-char/'],100*1e6,20,1000,'int8');


