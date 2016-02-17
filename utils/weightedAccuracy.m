function wAcc = weightedAccuracy( Y, Ypred , classFreq )
%WEIGHTEDACCURACY works for +-1 coding
%   Detailed explanation goes here

    t = size(Y,1);
%     classCoeffs = ((1 - classFreq)/sum(1 - classFreq)) / min((1 - classFreq)/sum(1 - classFreq));
    classCoeffs = ((1 ./ classFreq) * max(classFreq));
%     classCoeffs = ((1 - classFreq)/sum(1 - classFreq)) / min(((1 - classFreq)/sum(1 - classFreq)));
    sampleClassIdx = cell(1,t);
    for i = 1:t
        [ ~ , sampleClassIdx{i} ] = find(Y(i,:) == 1);
    end    
    
    normFactor = sum(classCoeffs);

    C = transpose(bsxfun( @eq, Y, Ypred ));
    D = sum(C,2);
    E = D == t;
    
    % Reweight E 
    F = zeros(size(E,1),size(E,2));
    for i = 1:t
        F(sampleClassIdx{i}) = E(sampleClassIdx{i})*classCoeffs(i);
    end
    
    numCorrect = sum(F);
    wAcc = numCorrect / normFactor;   
end
