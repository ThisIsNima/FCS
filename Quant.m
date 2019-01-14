function quant = Quant(vars, level)
%
% This function quantizes continuous variables
%     - 'LEVELS':     number of Quant levels, defaults to logarithmic
%                     choice. 
% Example:
%
% X=rand(100,1);
%X_Quant=Quant(X,64)
% default values
sample_count = size(vars,1);
%if no levels are specified
if (level == -1)
    level = 2;
    if (sample_count >= 8)
        level = max(2,ceil(min(log2(sample_count),sqrt(sample_count/20))));
    end
end
quant = vars;
Q = (1:(level-1))./level;
T = quantile(vars(:,1),Q);
% remove duplicate bins
T = unique(T);
val=zeros(length(vars(:,1)),1);
for i = 1:length(T)
    idx=(val==0) & vars(:,1) <= T(i);
    val(idx)=i;
end
val(val==0)=(length(T)+1);
quant(:,1)=val;
