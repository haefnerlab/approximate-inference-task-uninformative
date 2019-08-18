function resp=pred_ideal_observer(eps_tone,eps_vis,theta)
% Code for generating ideal observer responses for probability of reporting right
% Takes in auditory tone position, right visual cue position and the
% parameters (theta) as input. Theta is an array of length two containing the
% lapse rate and the effective auditory eccentricity


eps_tone=eps_tone(:)';
eps_vis=eps_vis(:)';
pr_lapse=theta(:,1);
sig_aud=theta(:,2);
resp=pr_lapse.*0.5+(1-pr_lapse).*normcdf(0,-eps_tone,sqrt(sig_aud));
end