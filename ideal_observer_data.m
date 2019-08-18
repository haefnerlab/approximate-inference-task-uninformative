%% Code for generating best fits for the ideal observer

%%
%Author: Sabyasachi Shivkumar


load exp_data
nsubj=size(prob_right_central,1);   %Number of Subjects
ntrials=40;                         %Number of trials 
fits_x=linspace(-10,10,201);        %Auditory tone positions for the ideal observer

for ll=1:nsubj
    
    param_lb=[0,1e-4];             %Lower bound for parameter
    param_ub=[1,1e4];              %Upper bound for parameter
    theta0=param_lb+(param_ub-param_lb).*rand(size(param_lb));     %Probability of lapse Auditory Uncertainty
    
    fx= @(theta) -1*sum(log(eps+binopdf(repmat(round(ntrials*prob_right_central(ll,:)),size(theta,1),1),ntrials,pred_ideal_observer(tone_position,0*tone_position,theta))),2)+ ...
        -1*sum(log(eps+binopdf(repmat(round(ntrials*prob_right_matched(ll,:)),size(theta,1),1),ntrials,pred_ideal_observer(tone_position,tone_position,theta))),2);
                                  %Log Likelihood over parameters with
                                  %uniform prior
    theta=bads(fx,theta0,param_lb,param_ub);                         %Finding maximum likelihood parameters
    fit_central(ll,:)=pred_ideal_observer(fits_x,0*fits_x,theta);    %Best fit for central condition
    fit_matched(ll,:)=pred_ideal_observer(fits_x,0*fits_x,theta);    %Best fit for matched condition
end

%Rearranging data into compact form
data_x=tone_position;
data_raw=reshape(prob_right_central,[nsubj,1,size(prob_right_central,2)]);
data_raw(:,2,:)=reshape(prob_right_matched,[nsubj,1,size(prob_right_matched,2)]);

fits=reshape(fit_central,[nsubj,1,size(fit_central,2)]);
fits(:,2,:)=reshape(fit_matched,[nsubj,1,size(fit_matched,2)]);

clearvars -except data_x data_raw fits fits_x

% Uncomment for saving
%save('ideal_observer.mat');