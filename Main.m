%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                              READ ME!                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                    %
%  1\ Run 'Main.m' to test the scripts.                              %
%  2\ Command window shows the progress and some accuracy values.    %
%  3\ For testing purpose, you may run one or all scripts of Q1~Q4   %
%     in 'Main.m' script by adding or removing a "%" in front of     %
%     the corresponding Qi(i=1,2,3,4).                               %
%  4\ Q1, Q2, Q3 and Q4 will generate 8 figures in total, but all    %
%     the figures won?t show until 'Main.m? finishing running.       %
%                                                                    %
%                                                                    %
%                                                                    %
%                                                 CHI JI (A0138141N) %
%                                                  Dept. of ECE, NUS %
%                                                            2015/11 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
fprintf('------------------PATTERN RECOGNITION------------------\n');
fprintf('Note: Figs won''t show until ''Main.m'' finishing running!\n\n');

% MUST RUN!!!
Q0_preprocess;

% TOTAL RUN TIME: ABOUT (120+160+300+60) SECONDS
Q1_PCA; 
%Q2_NMF; 
%Q3_LDA;
%Q4_GMM;

fprintf('-----------------------ALL DONE!----------------------\n\n');
