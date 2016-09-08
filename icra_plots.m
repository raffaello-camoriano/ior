for c = imbClassArr

    % Test error comparison plots
    
        
%         for kk = 1: numAlpha
%             figure
%             hold on
%             h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(1).inc_rlsc_norec.bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
%                 'r' , 0.1 , 0 , 1 , '-');
%             h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(kk).inc_rlsc_yesrec.bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
%                 'b' , 0.1 , 0 , 1 , '-');
%             title({['Test Error for imbalanced class # ' , num2str(c)] ; ['\alpha = ', num2str(alphaArr(kk))]});
%             legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
%             xlabel('n_{imb}')
%             ylabel('Test Accuracy')
%             hold off    
%         end
        
        
        % Accuracy
%         for kk = 2: numAlpha
%             figure
%             hold on
%             h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(1).bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
%                 'r' , 0.1 , 0 , 1 , '-');
%             h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(kk).bestValAccBuf(:,c,1:min(maxiter,out(c,2)))), ...
%                 'b' , 0.1 , 0 , 1 , '-');
% %             title(['\alpha = ', num2str(alphaArr(kk))]);
%             legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
%             xlabel('n_{imb}')
%             ylabel('Test Accuracy')
%             hold off    
%         end
        
        
        % C == 28
        figure
          
        box on
        grid on
        hold on

        hold on
        h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(1).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c)), ...
            'r' , 0.1 , 0 , 1 , '-');
        h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(2).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c)), ...
            'b' , 0.1 , 0 , 1 , '-');
        h3 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(3).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c)), ...
            'g' , 0.1 , 0 , 1 , '-');
        h4 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(4).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c)), ...
            'y' , 0.1 , 0 , 1 , '-');
        h5 = bandplot(1:min(maxiter,out(c,2)),squeeze(results(5).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c)), ...
            'cyan' , 0.1 , 0 , 1 , '-');
        legend([h1,h2, h3, h4, h5],'\alpha = 0','\alpha = 0.25','\alpha = 0.5','\alpha = 0.75','\alpha = 1','Location','northeast','FontSize',12)
        xlabel('n_{imb}','FontSize',16)
        ylabel('Imbalanced Class Test Accuracy','FontSize',16)
        hold off    
        
        %% C ~!28, separate figures for accuracy section
        for kk = 2: numAlpha
            
            figure
                        
            box on
            grid on
            hold on
            
            a1 = squeeze(results(1).bestValAccBuf(:,c,1:min(maxiter,out(c,2))));
            b1 = squeeze(results(1).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c));
            c1 = (t*a1 - b1) / (t-1);
            
            a2 = squeeze(results(kk).bestValAccBuf(:,c,1:min(maxiter,out(c,2))));
            b2 = squeeze(results(kk).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c));
            c2 = (t*a2 - b2) / (t-1);
            
            h1 = bandplot(1:min(maxiter,out(c,2)),c1, ...
                'r' , 0.1 , 0 , 1 , '-');
            h2 = bandplot(1:min(maxiter,out(c,2)),c2, ...
                'b' , 0.1 , 0 , 1 , '-');
%             title(['\alpha = ', num2str(alphaArr(kk))]);
%             legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
            xlabel('n_{imb}','FontSize',16)
            ylabel('Balanced Classes Test Accuracy','FontSize',16)
            hold off    
        end
        
        %% C ~!28, joined figures for crossval section
        figure
        hold on
        for kk = 2: numAlpha
            
            a1 = squeeze(results(1).bestValAccBuf(:,c,1:min(maxiter,out(c,2))));
            b1 = squeeze(results(1).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c));
            c1 = (t*a1 - b1) / (t-1);
            
            a2 = squeeze(results(kk).bestValAccBuf(:,c,1:min(maxiter,out(c,2))));
            b2 = squeeze(results(kk).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c));
            c2 = (t*a2 - b2) / (t-1);
            
            subplot(2,2,kk-1,'XTick',0:50:min(maxiter,out(c,2)))
            grid on        
            box on
            h1 = bandplot(1:min(maxiter,out(c,2)),c1, ...
                'r' , 0.1 , 0 , 1 , '-');
            h2 = bandplot(1:min(maxiter,out(c,2)),c2, ...
                'b' , 0.1 , 0 , 1 , '-');
            title(['\alpha = ', num2str(alphaArr(kk))]);
%             title({['Test Acc.; Bal. !C' , num2str(c)] ; ['\alpha = ', num2str(alphaArr(kk))]});
%             legend([h1,h2],'Naive RRLSC','Recoded RRLSC','Location','southeast')
%             xlabel('n_{imb}','FontSize',16)
%             ylabel('Test Accuracy','FontSize',16)
            hold off    
        end
        
        %% C ~!28, overlapping figures for crossval section
        figure
%         axes1 = axes('Parent',figure,'XTick',[0 25 50 75 100]);

        hold on            
        grid on        
        box on
            
        a1 = squeeze(results(1).bestValAccBuf(:,c,1:min(maxiter,out(c,2))));
        b1 = squeeze(results(1).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c));
        c1 = (t*a1 - b1) / (t-1);
        
        h = zeros(1,5);
        for kk = 2: numAlpha
            
            a2 = squeeze(results(kk).bestValAccBuf(:,c,1:min(maxiter,out(c,2))));
            b2 = squeeze(results(kk).bestCMBuf(:,c,1:min(maxiter,out(c,2)), c, c));
            c2 = (t*a2 - b2) / (t-1);

            h(kk) = plot(1:min(maxiter,out(c,2)),mean(c2),'LineWidth',1.5);

        end
        
        h(1) = bandplot(1:min(maxiter,out(c,2)),c1, ...
            'r' , 0.1 , 0 , 1 , '-');
        
        hold off    
        legend([h(1), h(2), h(3), h(4), h(5)],'\alpha = 0','\alpha = 0.25','\alpha = 0.5','\alpha = 0.75','\alpha = 1',...
            'Location','southeast','FontSize',12)
        xlabel('n_{imb}','FontSize',16)
        ylabel('Balanced Classes Test Accuracy','FontSize',16)


        %%

%         figure
%         hold on
%         h1 = bandplot(1:min(maxiter,out(c,2)),squeeze(results.inc_rlsc_norec.trainTime(:,c,1:min(maxiter,out(c,2)))), ...
%             'r' , 0.1 , 0 , 1 , '-');
%         h2 = bandplot(1:min(maxiter,out(c,2)),squeeze(results.inc_rlsc_yesrec.trainTime(:,c,1:min(maxiter,out(c,2)))), ...
%             'b' , 0.1 , 0 , 1 , '-');
%         title(['Training Time for imbalanced class # ' , num2str(c)]);
%         legend([h1,h2],'Na\"{i}ve RRLSC','Recoded RRLSC','Location','southeast')
%         xlabel('n_{imb}')
%         ylabel('Training Time')
%         hold off    
end