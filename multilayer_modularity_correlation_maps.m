clear all
close all
clc

addpath("startup+code")
addpath("/Users/haily/Documents/projects/hcp100ur")
%% set flags to 1 to make figure
plotflag(1) = 0;    % draw correlation matrices for scans 1 and 2
plotflag(2) = 0;    % draw multilayer communities on surfaces
plotflag(3) = 0;    % draw flexibility

%% load in hcp data from the 100 unrelated subjects
% the function below interacts w/the newly organized data. but you can get
% data from the same set of subjects by reading in the subjects in the
% 'hcp100ur.txt' file.
subj = 'all';
parc = '*schaefer400*';
task = '*REST*';
proc = 'regress_aCC24'; % can try 'regress_36pNS' for data with GSR
subcrtx = 'S3';

filename = 'correlation_mats_REST1_REST2_noGSR.mat';
check = dir(filename);

%% get fc
disp('getting fc')
if isempty(check)
    
    [ts,motion,info] = fcn_load_hcp(subj,parc,task,proc,subcrtx);
    
    % remove any subjects that have missing scans
    keep = sum(cellfun(@isempty,info),2) == 0;
    motion = motion(:,keep,:);
    ts = ts(:,:,keep,:);
    info = info(keep,:);
    % frame censoring
    thr = 0.15; % motion threshold
    [nt,n,nsub,nscan] = size(ts);
    rho = zeros(n,n,nsub,2);
    motionstats = zeros(nsub,2,3);
    
    % loop over subjects
    for isub = 1:nsub
        r = nan(n,n,nscan);
        ms = zeros(nscan,3);
        
        % loop over scans
        for iscan = 1:nscan
            
            % get low motion mask
            mask = motion(:,isub,iscan) < thr;
            
            % calculate fc
            r(:,:,iscan) = corr(ts(mask,:,isub,iscan));
            ms(iscan,1) = mean(motion(mask,isub,iscan));  % average subthr motion
            ms(iscan,2) = sum(mask);                      % number of frames
            ms(iscan,3) = mean(motion(~mask,isub,iscan)); % average suprathr motion
        end
        rho(:,:,isub,1) = nanmean(r(:,:,1:2),3);
        rho(:,:,isub,2) = nanmean(r(:,:,3:4),3);
        motionstats(isub,1,:) = nanmean(ms(1:2,:));
        motionstats(isub,2,:) = nanmean(ms(3:4,:));
    end
    % load behavioral data
    load behav_data.mat
    subj = cell(size(info,1),1);
    for isub = 1:size(info,1)
        str = strsplit(info{isub,1}{1},'.');
        subj{isub} = str{1};
    end
    subj = str2double(subj);
    subj = subj';
    % extract behavioral measures for the HCP100UR subjects
    [~,idx,~] = intersect(behav.subject,subj);
    tbl = behav(idx,:);
    names = {'Friend','Stress','Reject','Hostil','Emot','Lonli','Instr'};
    vars = [tbl.Friendship_Unadj,tbl.PercStress_Unadj,tbl.PercReject_Unadj,tbl.PercHostil_Unadj,tbl.EmotSupp_Unadj,tbl.Loneliness_Unadj,tbl.InstruSupp_Unadj];
    % loop over scans (REST1 and REST2)
    for iscan = 1:2
        
        % vectorize fc
        %load data/hcp400
        load hcp400
        X = zeros(n*(n - 1)/2,size(rho,3));
        mask = triu(ones(n),1) > 0;
        qc = zeros(max(lab),size(rho,3));
        for i = 1:size(rho,3)
            a = rho(:,:,i,iscan);
            X(:,i) = a(mask);
        end
        
        % fisher transform
        X = fcn_fisher(X);
        
        % regress out motion from each edge
        for i = 1:size(X,1)
            y = X(i,:)';
            m = [ones(length(y),1),squeeze(motionstats(:,iscan,1:2))]; % we include mean subthreshold motion and # of retained frames, but should also include ICV, age, etc. in final analysis
            [~,~,X(i,:)] = regress(y,m);
        end
        
        % compute correlations with all behavioral variables
        [R,P] = corr(X',vars,'type','spearman');
        S=[];
        S(:,:,iscan) = R; % retain the correlation patterns for later
        
        % make some plots
        if plotflag(1)
            sf = 0.25;
            m = str2double(strrep(strrep(parc,'schaefer',''),'*',''));
            load(sprintf('hcp%1.3i.mat',m));
            lab = [lab; ones(n - length(lab),1)*17];
            [gx,gy,idxlab] = fcn_plot_blocks(lab);
            for i = 1:size(R,2)
                mat = zeros(n);
                mat(mask) = R(:,i);
                mat = mat + mat';
                subplot(2,size(R,2),i + (iscan - 1)*size(vars,2));
                imagesc(mat(idxlab,idxlab),[-1,1]*sf);
                hold on;
                plot(gx,gy,'k');
                colormap(fcn_cmapjet);
                title(names{i});
                axis square;
            end
        end
    end
    save(filename,'S');
    save('corrsp_200.mat', 'P');
else
    load(filename);
end
%%
m = size(S,1);
n = (1 + sqrt(1 + m*8))/2;

%% calculate average correlation pattern across both scans
disp('calculating average correlation')
%addpath('fcn');

% number of nodes in cortex -- may need to change
ncortex = 400;

% calculate average
T = nanmean(S,3);

% set gamma value (here, we use the "potts" model -- gamma falls within
% range of [-1,1] but should probably be restricted to > 0.
gammas = linspace(0,0.9,19);
% linspace for gamma values
% interpretation is that mean connectivity weight of communities maximally exceed a correlation of "gamma"
% try to identify reasonable upper limit for gamma

% initialize multilayer tensor (flattened into 2d)
disp('initializing multilayer tensor')
Bs = zeros(n*size(T,2),n*size(T,2),length(gammas));
for g = 1:length(gammas)
    B = zeros(n*size(T,2));
    for i = 1:size(T,2)
        idx = (1:n) + (i - 1)*n;
        mat = zeros(n);
        mat(triu(ones(n),1) > 0) = T(:,i);
        mat = mat + mat';
        mat = corr(mat);
        b = (mat - gammas(g).*~eye(n));
        B(idx,idx) = b;
    end
    Bs(:,:,g) = B;
end
% add interlayer links -- here we link all layers to one another
omegas = logspace(1,5,50)*(1/1.0e+05); % log spacing of omegas
all2all = n*[(-size(T,2)+1):-1,1:(size(T,2)-1)];
% cirs = zeros(size(mat,1),size(T,2),length(omegas),length(gammas));
% flx = zeros(size(mat,1),length(omegas),length(gammas));
% numiter = 100;
% cicons = [];
% cis = zeros(size(mat,1)*size(T,2),numiter,length(omegas),length(gammas));
% agreement = zeros(n,n,size(T,2),length(omegas),length(gammas));
% aris = zeros(numiter,numiter,size(T,2),length(omegas),length(gammas));
% part = zeros(n,size(T,2),numiter,length(omegas),length(gammas));
% for the big run
cirs = zeros(size(mat,1),size(T,2));
flx = zeros(size(mat,1),1);
numiter = 100;
cicons = [];
cis = zeros(size(mat,1)*size(T,2),numiter);
agreement = zeros(n,n,size(T,2));
aris = zeros(numiter,numiter,size(T,2));
part = zeros(n,size(T,2),numiter);

% loop through omegas
for o = 1:length(omegas)
    omega = omegas(o);
    %disp(omega)
    
    %loop through gammas
    for g = 1:length(gammas)
        B_ = Bs(:,:,g) + omega*spdiags(ones(n*size(T,2),2*size(T,2)-2),all2all,n*size(T,2),n*size(T,2));
        %disp(gammas(g))
        
        for i = 1:100
            cis(:,i) = fcn_community_unif(B_); % could also use genlouvain -> "ci = genlouvain(B_);"
            cis(:,i) = fcn_sort_communities(cis(:,i));
        end

        % get consensus communities
        %cicon = fcn_consensus_communities(cis(:,:,o,g),numiter,true);
        %cicons = [cicons, cicon];

        % reshape communities
        %cirs(:,:,o,g) = reshape(cicon,[n,size(T,2)]);

        % visualize each layer's community structure
        %cmap = distinguishable_colors(max(cicon));
        %for i = 1:size(T,2)
        %    cols = cmap(cirs(:,i,o,g),:);
        %    cols = cols(1:ncortex,:);
        %    if plotflag(2)
        %        f = fcn_surfquad(1:ncortex,cols,[1,ncortex],true,0.25,'yeo');
        %        figname = num2str(omegas(o)) + "_" + num2str(gammas(g)) + "_" + num2str(i) + ".png";
        %        saveas(f, figname)
        %        close(gcf)
        %    end
        %end
        
        % get degeneracy of partition landscape across runs of community
        % detection algorithm
        a = reshape(cis(:,:),[n,size(T,2),numiter]);
        part(:,:,:) = a;
        % one layer at a time
        for i = 1:size(T,2)
            ci = squeeze(a(:,i,:));
            % calculate aris of partitions
            aris(:,:,i) = fcn_ari_fast(ci);
        end

        % calculate flexibility (how frequently a node's community assignment
        % varies across layers -- this code compares all pairs of layers (not
        % appropriate for time-varying networks)
        % need to update to include gammas
        flx(:,1) = fcn_fastflx(cirs(:,:));
        if plotflag(3)
            % if all flx values are 0, which is true when omega is large,
            % then interp1 will return an error
            if max(flx(:,o,g)) > 0
                cols = interp1(linspace(min(flx(:,o,g)),max(flx(:,o,g)),256),jet(256),flx(1:ncortex,o,g));
                f = fcn_surfquad(1:ncortex,cols,[1,ncortex],true,0.25,'yeo');
                figname = num2str(omegas(o)) + "_" + num2str(gammas(g)) + "_flex.png";
                saveas(f, figname)
                close(gcf)
                cols = interp1(linspace(min(mu),max(mu),256),jet(256),mu(1:ncrtx,:));
            end
        end
        flex_filename = append('flex_o',num2str(omegas(o)),'_g',num2str(gammas(g)),'.mat');
        ari_filename = append('aris_o',num2str(omegas(o)),'_g',num2str(gammas(g)),'.mat');
        part_filename = append('part_o',num2str(omegas(o)),'_g',num2str(gammas(g)),'.mat');
        disp(part_filename)
        save(flex_filename, 'flx');
        save(ari_filename, 'aris');
        save(part_filename, 'part');
    end
end

%% calculate variability in flexibility and average
%va = zeros(length(omegas),length(gammas));
%for o = 1:length(omegas)
%for o = 1:9 
    %for g = 1:length(gammas)
%    for g = 1:19
%        va(o,g) = var(flx(:,o,g));
%    end
%end

%avg = zeros(length(omegas),length(gammas));
%for o = 1:length(omegas)
%    for g = 1:length(gammas)
%        avg(o,g) = mean(flx(:,o,g));
%    end
%end

%% save slices for analysis in R
% top 3 most variable: gamma = 0.05, omega = 0.01 (var = 0.1062);
% gamma = 0.05, omega = 0.0003 (var = 0.1038)
% gamma = 0.3, omega = 0.001 (var = 0.1058)
%writematrix(flx(:,5,2), 'flex_var0.1062.csv')
%writematrix(cirs(:,:,5,2), 'multicommlabels_var0.1062.csv')
%writematrix(flx(:,2,2), 'flex_var0.1038.csv')
%writematrix(cirs(:,:,2,2), 'multicommlabels_var0.1038.csv')
%writematrix(flx(:,7,3), 'flex_var0.1058.csv')
%writematrix(cirs(:,:,7,3), 'multicommlabels_var0.1058.csv')
% slice that has max magnitude of expression of pc1
% gamma = 0.15 (3rd), omega = 0.001 (3rd)
%writematrix(cirs(:,:,3,3), 'comlabelsPC1.csv')
%writematrix(flx(:,3,3), 'flexPC1.csv')

%% save everything
disp('saving')
save('flex_socsup.mat', 'flx'); % nodel level flexibility across seven layers given consensus community labels
save('multilayercommlabels.mat', 'cirs'); % multilayer community labels
save('aris_bigrun.mat', 'aris'); % aris within layer of all iterations of multilayer community detection
save('flex_var.mat', 'va'); % variability of flexbility across seven layers
save('partlandscape_bigrun.mat', 'part'); % partition landscape
writematrix(vars, 'socsuppvars.csv')

%% single-layer example

% layer index
%idxlayer = 1;

% create matrix
%mat = zeros(n);
%mat(triu(ones(n),1) > 0) = T(:,idxlayer);
%mat = mat + mat';

% observed - expected
%gamma = 0.0;
%b = (mat - gamma).*~eye(n);

% community detection
%ci = fcn_community_unif(b);
%ci = fcn_sort_communities(ci);

% draw communities
%cmap = distinguishable_colors(max(ci));
%cols = cmap(ci,:);
%cols = cols(1:ncortex,:);
%f = fcn_surfquad(1:ncortex,cols,[1,ncortex],true,0.25,'yeo');


% get correlations of scores on social support measures
imagesc(corr(vars))
colorbar
colormap('jet')
% get correlations of correlations between edge weights and social support
% measures across the two rest scans
imagesc(corr(S(:,:,1),S(:,:,2)))
colorbar
colormap('jet')

