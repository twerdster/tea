function genTest_Preloaded()
testStr = 'T8';
default=struct('numF',200,'preload',20,'depth',10,'fold',10,'Ftype','F_CHAR','numS',10*1e6,'numT',50,'numGPU',1,'weightType','W_ONES','baseDir','','prefix','','log','LOG2','comment','');

%Speed:
%T8: Need to test number of features to preload:10-200  (#preload=10,20,50,100,200 with Odem, #GPUS=0,1,2,3, #samples=100m, depth=20 ) = (20 tests)
% Must show with and without caching
% Can show that when memory bound having multiple GPUs doesnt help
% Can show that earlier stages are memory bound, later are not and
% preloading allows latency hiding
% can illustrate the time it takes according to the read speed
% Should only show on one gpu I think, but can do everyone anyway

baseTestDir='200-54-7-char';
preloads=[4,20,50,100, 200];
dropCache=[1];
machines= { ...
    'k5000m',  '/media/gipadmin/data/' ,  ...
    'k80',     '/home/aaron/forks/tea/data/'    ,  ...
    'titanz',  '/home/gipuser/forks/tea/data/'  ,  ...
    'gtx1080', '/home/aaron/forks/tea/data/'    ,  ...
    'gtx580',  '/home/admin/forks/tea/data/'    ,  ...
    };
gpus = containers.Map;
gpus('k5000m')=[1];
gpus('k80')=[1 2];
gpus('titanz')=[1 2];
gpus('gtx1080')=[1 2 3];
gpus('gtx580')=[1 2 4];

T=default;
%% Constant test component
% --------------------------------
T.numF=200;
T.depth=16;
T.fold=16;
T.numS=54*1e6;
T.numT=10; % The threshold checks define time taken doing processing which changes relevance of preloaded.
T.comment='Test: Timing preload values with and without disk buffering on different numbers of GPUs';
% --------------------------------
for i = 2:2:length(machines)
    machine = machines{i-1};
    T.prefix=['-' machine '-' testStr];
    baseDir = [machines{i} baseTestDir ];
    T.baseDir=baseDir;
    f=fopen(sprintf('%s-%s.test',machine,testStr),'w');
    fprintf('%s\n',T.comment);
    fprintf('Machine: %s\n',machine);
    for preload = preloads
        for numGPU = gpus(machine)
            for dc = dropCache
                %% Dynamic test component
                % --------------------------------
                T.numGPU=numGPU;
                T.preload=preload;
                fprintf('%s\n',genTestString(T,dc));
                fprintf(f,'%s\n',genTestString(T,dc));
                % --------------------------------
            end
        end
    end
    fprintf('\n');
    fclose(f);
end


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




