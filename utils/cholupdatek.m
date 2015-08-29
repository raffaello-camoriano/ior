function [ R ] = cholupdatek( R , X )
%CHOLUPDATEK Rank-k Cholesky update helper function

    for i = 1:size(X,1)
        R = cholupdate(R,X(i,:)');                
    end
end

