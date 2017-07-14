function genTest_NumClasses()
testStr = 'T12';
default=struct('numF',200,'preload',20,'depth',10,'fold',10,'Ftype','F_CHAR','numS',10*1e6,'numT',50,'numGPU',1,'weightType','W_ONES','baseDir','','prefix','','log','LOG2','comment','');

%Speed:
%T12: Need to do a couple of extreme tests with the full monty i.e. 2bn and 500 features, 600m and 2000 features, 100m and 10000 features  (lets say this is 10 tests)
machines= { ...
    'k5000m',  '/media/gipadmin/data/' ,  ... %2
    'k80',     '/home/aaron/forks/tea/data/'    ,  ... %4
    'titanz',  '/home/gipuser/forks/tea/data/'  ,  ... %6
    'gtx1080', '/home/aaron/forks/tea/data/'    ,  ... %8
    'gtx580',  '/home/admin/forks/tea/data/'    ,  ... %10
    };
gpus = containers.Map;
gpus('k5000m')=[1];
gpus('k80')=[1 2];
gpus('titanz')=[1 2];
gpus('gtx1080')=[1 2 3];
gpus('gtx580')=[1 2 3 4];


%% --------------------------------
T=default;
T.comment='Test: Full 2bn, 20 deep,';
T.depth=20;
T.fold=20;
T.numF=100;
T.numS=2000*1e6;
T.numT=50;
dropCache=false;
i=8;
machine = machines{i-1};
T.numGPU=3;
T.prefix=['-' machine '-' testStr];
f=fopen(sprintf('%s-%s-0.test',machine,testStr),'w');
fprintf('%s\n',T.comment);
fprintf('Machine: %s\n',machine);
T.baseDir=[machines{i} '100-2000-32-char']  ;
fprintf('%s\n',genTestString(T,dropCache));
fprintf(f,'%s\n',genTestString(T,dropCache));
fprintf('\n');
fclose(f);

%% --------------------------------
T=default;
T.comment='Test: Full 600m, 20 deep, 2000f';
T.depth=20;
T.fold=20;
T.numF=2000;
T.numS=600*1e6;
T.numT=50;
dropCache=false;
i=10;
machine = machines{i-1};
T.numGPU=4;
T.prefix=['-' machine '-' testStr];
f=fopen(sprintf('%s-%s-1.test',machine,testStr),'w');
fprintf('%s\n',T.comment);
fprintf('Machine: %s\n',machine);
T.baseDir=[machines{i} '100-600-32-char']  ;
fprintf('%s\n',genTestString(T,dropCache));
fprintf(f,'%s\n',genTestString(T,dropCache));
fprintf('\n');
fclose(f);

%% --------------------------------
T=default;
T.comment='Test: Full 10m, 20 deep, 10000f';
T.depth=20;
T.fold=20;
T.numF=10000;
T.numS=10*1e6;
T.numT=50;
dropCache=false;
i=8;
machine = machines{i-1};
T.numGPU=3;
T.prefix=['-' machine '-' testStr];
f=fopen(sprintf('%s-%s-2.test',machine,testStr),'w');
fprintf('%s\n',T.comment);
fprintf('Machine: %s\n',machine);
T.baseDir=[machines{i} '100-600-32-char']  ;
fprintf('%s\n',genTestString(T,dropCache));
fprintf(f,'%s\n',genTestString(T,dropCache));
fprintf('\n');
fclose(f);

function testPrefix = genTestPrefix(T)
testPrefix = sprintf('%u-%u-%u-%u-%s-%u-%u-%u-%s', ...
    T.numF, ...
    T.preload, ...
    T.depth, ...
    T.fold, ...
    T.Ftype, ...
    T.numS, ...
    T.numT, ...
    T.numGPU, ...
    T.weightType);


function testStr = genTestString(T,dropCache)
dc='';
if exist('dropCache') && dropCache
    dc='sudo DROP_CACHE=1 ';
end

testStr = sprintf('%s./Tea %u %u %u %u %s %u %u 0 %u %s %s/ %s 0 %s "%s"', ...
    dc, ...
    T.numF, ...
    T.preload, ...
    T.depth, ...
    T.fold, ...
    T.Ftype, ...
    T.numS, ...
    T.numT, ...
    T.numGPU - 1, ...
    T.weightType, ...
    T.baseDir, ...
    [genTestPrefix(T) T.prefix], ...
    T.log, ...
    T.comment);

