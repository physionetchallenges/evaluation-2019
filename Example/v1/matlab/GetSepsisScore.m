function [scores, labels] = GetSepsisScore(input_file)

% read the data
Ta = ReadChallengeData(input_file);
% grab the features
X=Ta(:,1:34);       [Xd1,Xd2] = size(X);
COVAR=Ta(:,35:40);  [COVARd1,COVARd2] = size(COVAR);
% normalization
Xnorm_info.mean= [83.8996 97.0520   36.8055  126.2240   86.2907 66.2070   18.7280   33.7373   -3.1923   22.5352    0.4597  7.3889 ...
    39.5049   96.8883  103.4265   22.4952   87.5214    7.7210  106.1982    1.5961    0.6943 131.5327    2.0262    2.0509    3.5130   ...
    4.0541    1.3423    5.2734   32.1134   10.5383   38.9974   10.5585  286.5404  198.6777];
Xnorm_info.std = [17.6494    3.0163    0.6895   24.2988   16.6459   14.0771    4.7035   11.0158    3.7845    3.1567    6.2684   ...
    0.0710    9.1087    3.3971  430.3638   19.0690   81.7152    2.3992    4.9761    2.0648    1.9926  45.4816    1.6008    0.3793 ...
    1.3092    0.5844    2.5511   20.4142    6.4362    2.2302   29.8928    7.0606  137.3886   96.8997];
Xcovnorm_info.mean = [60.8711 0.5435 0.0615 0.0727 -59.6769 28.4551];
Xcovnorm_info.std = [16.1887 0.4981 0.7968 0.8029 160.8846 29.5367];

X = (X - repmat(Xnorm_info.mean,Xd1,1))./repmat(Xnorm_info.std,Xd1,1);
COVAR = (COVAR - repmat(Xcovnorm_info.mean,COVARd1,1))./repmat(Xcovnorm_info.std,COVARd1,1);
for k=1:Xd2, X(isnan(X(:,k)),k)=0;end
for k=1:COVARd2, COVAR(isnan(COVAR(:,k)),k)=0;end
% model
model.beta = [ 0.1806    0.0249    0.2120   -0.0495    0.0084   -0.0980    0.0774   -0.0350   -0.0948    0.1169    0.7476    ...
    0.0323    0.0305   -0.0251    0.0330    0.1424    0.0324   -0.1450   -0.0594    0.0085   -0.0501 0.0265    0.0794   -0.0107 ...
    0.0225    0.0040    0.0799   -0.0287    0.0531   -0.0728    0.0243    0.1017    0.0662   -0.0074    0.0281    0.0078  ...
    0.0593   -0.2046   -0.0167    0.1239]';
model.rho = 7.8521;
model.nu = 1.0389;

scores = 1 - wc_survival(4,[X COVAR],model);
labels = double([scores>0.45]);
end


function S = wc_survival(t,xstar,model)

expBX = exp(xstar*model.beta);
Lambda = (t/model.rho).^model.nu;
LexpBX = Lambda.*expBX;
S = exp( - LexpBX );

end


function [values, column_names] = ReadChallengeData(filename)
  f = fopen(filename, 'rt');
  try
    l = fgetl(f);
    column_names = strsplit(l, '|');
    values = dlmread(filename, '|', 1, 0);
  catch ex
    fclose(f);
    rethrow(ex);
  end
  fclose(f);

  %% ignore SepsisLabel column if present
  if strcmp(column_names(end), 'SepsisLabel')
    column_names = column_names(1:end-1);
    values = values(:,1:end-1);
  end
end
